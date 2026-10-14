from itertools import product
import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from suc_benders.benders import coordinate, dispatch, master
from suc_benders.config import ROOT, load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.qubo import qubo
from suc_benders.solver import solve
from suc_benders.validation import pack, repair, unpack, validate


@pytest.mark.parametrize("name,shape,nnz,fixed_cost,relaxed", [
    ("deterministic", (12648, 3840), 30636, 1223143.1202172905, 815439.9035949688),
    ("stochastic", (21960, 6912), 56996, 1220423.7732877277, 841376.5248539273)])
def test_rts_model_reference(name, shape, nnz, fixed_cost, relaxed):
    fixture = ROOT / "tests/fixtures/rts24"
    d, s = load_data(fixture / "units.json", fixture / f"{name}.json")
    m = model(d, s)
    assert m.A.shape == shape and m.A.nnz == nnz
    # Digests come from the archived MATLAB model, not the Python builder.
    digests = {"deterministic": "c609e14bb688a225b4427d1cbc7e5930dd8f3ad41c2415f175e41a6143934faa",
               "stochastic": "c4b25b83e2ee69e95c61e76aecd4b3766311597d03a3d43a6e0e9ad3f1d9f747"}
    a = m.A.copy(); a.sort_indices(); checksum = hashlib.sha256()
    for v in (a.indptr, a.indices):
        checksum.update(np.asarray(v, dtype="<i8").tobytes())
    for v in (a.data, m.rhs, m.eq, m.obj, m.q, m.lb, m.ub):
        checksum.update(np.asarray(v, dtype="<f8").ravel().tobytes())
    assert checksum.hexdigest() == digests[name]
    assert d.pmax.sum() == 3405
    r = optimize(m, 30)
    assert r.status == "OPTIMAL"
    assert abs(r.objval - relaxed) < .05
    assert r.lower <= r.objval + 1e-5
    full = partition(m, d.horizon)[0][0]
    fixed = dispatch(full, repair(d, s, np.ones((d.n_units, d.horizon))), 30)
    assert fixed.feasible and abs(fixed.cost - fixed_cost) < 1e-3
    assert validate(d, s, unpack(m, fixed.point))["feasible"]


@pytest.mark.parametrize("hours", [1, 2, 3, 5])
def test_partition_identity(tiny, hours):
    d, s = tiny(2, 5); d.min_up[:] = 4; d.min_down[:] = 3
    m = model(d, s); rng = np.random.default_rng(1)
    point = rng.random(len(m.obj)); prices = rng.random(len(m.rhs)); prices[m.eq] *= -1
    blocks, cross = partition(m, hours, prices)
    rows, columns, cost, constants = list(cross), [], 0., np.zeros(len(cross))
    for b in blocks:
        rows.extend(b.rows); columns.extend(b.cols)
        p = point[b.cols]
        cost += b.obj @ p + b.q @ (p ** 2) + b.constant
        constants += b.crossrhs
    np.testing.assert_array_equal(sorted(rows), np.arange(len(m.rhs)))
    np.testing.assert_array_equal(sorted(columns), np.arange(len(m.obj)))
    np.testing.assert_array_equal(constants, m.rhs[cross])
    original = m.obj @ point + m.q @ (point ** 2) + prices[cross] @ (m.A[cross] @ point - m.rhs[cross])
    assert abs(cost - original) < 1e-8
    assert blocks[-1].times[-1] == 4


def test_coordinate_selects_small_multipliers_on_flat_optimal_face(monkeypatch):
    from suc_benders import benders
    original = benders.optimize
    calls = []

    def degenerate(m, seconds):
        result = original(m, seconds)
        if not calls:
            # The second multiplier has zero slope in every plane, so this remains optimal.
            result.x[1] = 1e18
        calls.append(result)
        return result

    monkeypatch.setattr(benders, "optimize", degenerate)
    planes = [[(5., np.array([-1., 0.])), (0., np.array([1., 0.]))]]
    prices, upper = coordinate(planes, np.array([False, True]), 10., 10.)
    assert len(calls) == 2 and calls[0].x[1] == 1e18
    assert upper == pytest.approx(2.5)  # Preserve the primary objective, not the L1 objective.
    assert prices[0] == pytest.approx(2.5, abs=2e-5) and abs(prices[1]) < 1e-8
    assert min(c+slope@prices for c, slope in planes[0]) >= upper-2e-5


def test_cuts_exhaustive_and_lp_bound(tiny):
    d, s = tiny(); s[0].load[:] = [30, 0, 0]
    m = model(d, s); b = partition(m, 3)[0][0]
    records, cuts = [], []
    for bits in product([0., 1.], repeat=6):
        x = pack(d, np.asarray(bits).reshape((2, 3), order="F"))
        r = dispatch(b, x, 10)
        if r.cut is not None:
            cuts.append(r.cut); records.append((x, r))
            value = r.cut.alpha + r.cut.beta @ x
            if r.feasible:
                assert abs(value - r.recourse) < 2e-5
            else:
                assert value > 1e-6
    assert any(c.optimality for c in cuts) and any(not c.optimality for c in cuts)
    cost = min(r.cost for x, r in records if r.feasible)
    for x, r in records:
        if r.feasible:
            for c in cuts:
                assert c.alpha + c.beta @ x <= (r.recourse if c.optimality else 0) + 1e-5
    relaxed_master, bound = master(b, cuts, 10)
    assert bound <= cost + 2e-4 and not hasattr(relaxed_master, "x")
    assert abs(optimize(m, 10, True).objval - cost) < 2e-5


def test_startup_shutdown_capacity(tiny):
    d, s = tiny(1); s[0].load[:] = [35, 0, 0]
    x = pack(d, np.asarray([[1., 0, 0]]))
    assert dispatch(partition(model(d, s), 3)[0][0], x, 10).feasible
    s[0].load[0] = 35.1
    r = dispatch(partition(model(d, s), 3)[0][0], x, 10)
    assert not r.feasible and r.cut.alpha + r.cut.beta @ x > 0


@pytest.mark.parametrize("row,rhs,equality,x,expected", [
    ([1., 0.], 0., False, 1., False),
    ([-1., 0.], 0., False, 1., True),
    ([-1., 0.], 0., True, 1., False),
    ([1., 0.], 1.+1e-8, True, 1., True),
    ([1., 0.], 1.+1e-6, True, 1., False),
    ([1., 1.], 0., False, 1., True),
    ([1., 1.], 0., False, .5, False),
])
def test_discrete_feasibility_ignores_dispatch_rows(row, rhs, equality, x, expected):
    from suc_benders.solver import discrete_feasible
    b = NS(A=sparse.csr_matrix([row]), rhs=np.array([rhs]), eq=np.array([equality]),
           binary=np.array([0]), continuous=np.array([1]))
    assert discrete_feasible(b, np.array([x])) is expected


def encoding_model(A, rhs, kind, scale=10, hi=4):
    return NS(A=sparse.csr_matrix(A, dtype=float), rhs=np.asarray(rhs, float), eq=np.array([False, False]),
              obj=np.array([0., 0., 1.]), q=np.zeros(3), lb=np.zeros(3), ub=np.array([1., 1., hi]),
              binary=np.arange(2), nx=2, and_pairs=[], kind=kind, scale=scale)


def test_qubo_full_encoding(options):
    m = encoding_model([[1, 1, 0], [1, 2, -1]], [1, 0], ["hard", "optimality"])
    q = qubo(m, options, 4)
    n = q.Q.shape[0]; assert n <= 12
    minimum = np.full((4, 5), np.inf)
    for bits in product([0., 1.], repeat=n):
        bits = np.asarray(bits); x = q.fixed.copy(); x[q.free] = bits[:len(q.free)]
        theta = int(round(q.theta_lower + q.theta_weights @ bits[q.theta_ids]))
        index = int(x[0] + 2 * x[1]); energy = bits @ q.Q @ bits + q.offset
        minimum[index, theta] = min(minimum[index, theta], energy)
    for code in range(4):
        x = np.array([code % 2, code // 2])
        for theta in range(5):
            if (m.A @ np.r_[x, theta] <= m.rhs).all():
                assert abs(minimum[code, theta] - theta / m.scale) < 1e-8
            else:
                assert minimum[code, theta] > theta / m.scale + 1


def test_fractional_capacity_encoding_builds(options):
    m = encoding_model([[-.15, -.25, 0], [.27, -.31, -1]], [-.2, -.18], ["capacity", "optimality"], 1, 1)
    options.theta_step = .1
    q = qubo(m, options, 1)
    assert q.Q.shape[0] > len(q.free)


def test_capacity_lattice_feasibility_exhaustive(options):
    m = encoding_model([[-.15, -.25, 0]], [-.20000001], ["capacity"], 1, 0)
    m.eq = np.array([False])
    q = qubo(m, options, 0)
    minimum = np.full(4, np.inf)
    for bits in product([0., 1.], repeat=q.Q.shape[0]):
        bits = np.asarray(bits)
        index = int(bits[0]+2*bits[1])
        minimum[index] = min(minimum[index], float(bits @ q.Q @ bits + q.offset))
    for code in range(4):
        feasible = .15*(code % 2)+.25*(code // 2) >= .20000001
        assert (abs(minimum[code]) < 1e-7) == feasible


def test_free_equality_price():
    prices, upper = coordinate([[(-2., np.array([-1.])), (2., np.array([1.]))]], np.array([True]), 10, 10)
    assert abs(prices[0] + 2) < 1e-7 and abs(upper) < 1e-7


@pytest.mark.parametrize("hours", [1, 2, 3, 5])
def test_tiny_lp_gap_and_validation(tiny, options, hours, monkeypatch):
    from suc_benders import solver
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    d, s = tiny(1)
    options.block_hours = hours
    r = solve(d, s, options)
    # Without integer certification this instance retains an LP integrality gap.
    assert r.status == "MAX_ITERATIONS", r
    assert len(r.outer_history) == options.max_outer
    assert all(row["unconverged_blocks"] for row in r.outer_history)
    assert abs(r.objective - 225) < 1e-5
    assert r.lower_bound <= r.objective + 1e-5
    assert validate(d, s, r)["feasible"]
    r.scenarios[0].p[0, 0] += 1
    with pytest.raises(ValueError):
        validate(d, s, r)


def test_deadline(tiny, options):
    d, s = tiny(1); options.time_limit = 1e-9
    r = solve(d, s, options)
    assert r.status == "TIME_LIMIT" and not np.isfinite(r.objective)


@pytest.mark.parametrize("change", [{"unknown": 1}, {"max_inner": 0}, {"time_limit": -1},
                                      {"llm_enabled": True, "llm_model": ""}, {"seed": True}])
def test_option_validation(change, tmp_path):
    with pytest.raises(ValueError):
        load_options(change, tmp_path / "missing.json")


def test_unlimited_llm_calls_option(tmp_path):
    assert load_options({"llm_max_calls": None}, tmp_path / "missing.json").llm_max_calls is None
