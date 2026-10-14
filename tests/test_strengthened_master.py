from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from suc_benders import strengthened_master as strong
from suc_benders.benders import dispatch, master
from suc_benders.model import model, partition
from suc_benders.optimization import continuous_only, optimize
from suc_benders.validation import pack


def case(tiny):
    d, scenarios = tiny(n=2, T=2)
    m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
    root, _ = master(full, [])
    root = strong.add_temporal(root, m)
    return m, full, root, strong.build(m, root, len(root.obj)-1)


def test_model_and_costs_preserve_all_tiny_feasible_schedules(tiny):
    m, full, root, lifted = case(tiny)
    costs = []
    for bits in product((0, 1), repeat=4):
        x = pack(m.d, np.asarray(bits).reshape(2, 2))
        actual = dispatch(full, x, 5)
        if not actual.feasible:
            continue
        costs.append(actual.cost)
        fixed = deepcopy(lifted); fixed.lb[m.binary] = x; fixed.ub[m.binary] = x
        raw = optimize(fixed, 5)
        assert raw.status == "OPTIMAL" and raw.objval <= actual.cost+1e-5
        gap, cuts = strong.perspective_batch(m, fixed, raw.x, 1e-8)
        raw2 = optimize(strong.add_perspectives(fixed, m, cuts), 5)
        assert raw2.status == "OPTIMAL" and raw2.lower+1e-5 >= raw.lower
        assert raw2.objval <= actual.cost+1e-5
    refined = strong.separate(m, lifted, 5, .001)
    assert refined.status == "SEPARATION_CLOSED" and refined.gap <= .001
    certified = strong.certify(refined.model, 5)
    assert certified.optimizer_calls == 1 and certified.lower <= min(costs)+1e-5


@pytest.mark.parametrize("initial,minimum", list(product((0, 1), (1, 3))))
def test_cumulative_rows_preserve_legal_integer_paths(tiny, initial, minimum):
    d, scenarios = tiny(n=1, T=4)
    d.initial_on[:] = initial; d.initial_output[:] = d.pmin*initial
    d.initial_duration[:] = 3; d.min_up[:] = minimum; d.min_down[:] = minimum
    m = model(d, scenarios); full = partition(m, 4)[0][0]
    root, _ = master(full, []); strengthened = strong.add_temporal(root, m)
    for bits in product((0, 1), repeat=4):
        x = pack(d, np.asarray(bits)[None, :])
        hard = full.A[:, full.continuous].getnnz(axis=1) == 0
        r = full.A[hard][:, full.binary] @ x-full.rhs[hard]; r[full.eq[hard]] = abs(r[full.eq[hard]])
        if r.max(initial=0.) <= 1e-7:
            point = np.zeros(len(root.obj)); point[:len(x)] = x
            assert (strengthened.A[len(root.rhs):] @ point-strengthened.rhs[len(root.rhs):]).max(initial=0.) <= 1e-7


def test_invalid_cut_identity_start_and_mode_rejected(tiny):
    m, full, root, lifted = case(tiny)
    cut = strong.perspective_cut(m, 0, 0, 0, 20.)
    cut["u_coefficient"] -= 1.
    with pytest.raises(ValueError):
        strong.add_perspectives(lifted, m, [cut])
    changed = deepcopy(m); changed.rhs[0] += 1.
    with pytest.raises(ValueError, match="identity"):
        strong.perspective_batch(changed, lifted, np.zeros(len(lifted.obj)), .01)
    with pytest.raises(ValueError, match="MIP start"):
        strong.certify(lifted, 5, warm_start=np.full(len(lifted.obj), np.nan))
    with pytest.raises(ValueError, match="integer"):
        optimize(lifted, 5, mip_focus=3)
    with continuous_only(lambda event: None), pytest.raises(ValueError, match="forbidden"):
        strong.certify(lifted, 5)


def test_mip_start_and_bound_target(tiny):
    m, full, root, lifted = case(tiny)
    actual = dispatch(full, pack(m.d, np.ones((2, 2))), 5)
    warm = np.zeros(len(lifted.obj)); warm[:len(m.obj)] = actual.point
    for item in lifted.and_lift:
        warm[item["index"]] = actual.point[item["v"]]*actual.point[item["w"]]
    for p, eta in lifted.eta.items():
        warm[eta] = m.q[p]*actual.point[p]**2
    master_start = np.zeros(len(root.obj)); master_start[:root.nx] = actual.point[m.binary]
    for z, v, w in root.and_pairs:
        master_start[z] = master_start[v]*master_start[w]
    master_start[-1] = actual.cost
    warm[lifted.master_columns] = master_start
    result = strong.certify(lifted, 5, warm_start=warm, target_lower=0.)
    assert result.status in ("OPTIMAL", "USER_OBJ_LIMIT") and result.lower <= actual.cost
    assert result.predicted_objective is not None


def test_late_bound_not_scored(tiny, monkeypatch):
    m, full, root, lifted = case(tiny)
    ticks = iter((0., 10.))
    monkeypatch.setattr(strong, "perf_counter", lambda:next(ticks))
    monkeypatch.setattr(strong, "optimize", lambda *a, **k:type("Raw", (), dict(status="OPTIMAL", lower=1., x=None, objval=2.))())
    result = strong.certify(lifted, 1.)
    assert result.status == "TIME_LIMIT" and result.lower == -np.inf


def test_bad_theta_and_numerical_status_cannot_certify(tiny, monkeypatch):
    m, full, root, lifted = case(tiny)
    wrong = deepcopy(root); wrong.obj[:] = 0.; wrong.obj[0] = 1.
    with pytest.raises(ValueError, match="theta"):
        strong.build(m, wrong, 0)
    monkeypatch.setattr(strong, "optimize", lambda *a, **k:type("Raw", (), dict(status="NUMERIC", lower=1., x=None, objval=2.))())
    assert strong.certify(lifted, 1.).lower == -np.inf
