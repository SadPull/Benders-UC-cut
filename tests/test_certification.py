from itertools import product
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import certification, solver
from suc_benders.benders import dispatch
from suc_benders.config import load_options
from suc_benders.hourly_bounds import HourlyBounds
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import pack


def whole_day(tiny, options, n=2):
    d, scenarios = tiny(n)
    options.bound_mode = "integer_benders"
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    started = perf_counter()
    whole = solver.WholeDay(m, full, options, started+30, solver.Result(lower_bound=0.), started)
    assert whole.accept(pack(d, np.ones((n, d.horizon))), "initial")
    return whole


@pytest.mark.parametrize("n,hours", [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_certification_closes_integer_gap_after_sa_limit(tiny, options, monkeypatch, n, hours):
    d, scenarios = tiny(n)
    options.bound_mode = "integer_benders"
    options.bound_slice_seconds = .2
    options.max_outer = options.max_inner = 1
    options.block_hours = hours
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    result = solver.solve(d, scenarios, options)
    assert result.status == "CONVERGED", result.message
    assert result.validation["feasible"]
    assert result.objective == pytest.approx(225.)
    assert 225.-.02251 <= result.lower_bound <= 225.+1e-5
    assert result.relative_gap <= 1e-4 + 1e-7
    assert result.certification_history
    if n == 2:
        assert len(result.certification_history) > 1
        assert result.diagnostics["coordination_count"] == 1
    lowers = [r["lower"] for r in result.certification_history]
    assert lowers == sorted(lowers)
    assert {c["source"] for c in result.diagnostics["certificate_cuts"]} >= {"initial"}


def test_certified_cuts_are_valid_for_every_feasible_commitment(tiny):
    d, scenarios = tiny()
    scenarios[0].load[:] = [30, 0, 0]
    m = model(d, scenarios)
    full = partition(m, 3)[0][0]
    records, cuts = [], []
    for bits in product([0., 1.], repeat=6):
        x = pack(d, np.asarray(bits).reshape((2, 3), order="F"))
        r = dispatch(full, x, 5, certified=True)
        if r.cut is not None:
            assert r.cut.certified
            cuts.append(r.cut)
            if not r.feasible:
                assert r.cut.alpha + r.cut.beta @ x > 1e-7
        if r.feasible:
            records.append((x, r))
    assert any(c.optimality for c in cuts) and any(not c.optimality for c in cuts)
    for x, r in records:
        for c in cuts:
            assert c.alpha + c.beta @ x <= (r.recourse if c.optimality else 0.) + 1e-7
    assert min(r.cost for _, r in records) == pytest.approx(optimize(m, 5, integral=True).objval)


def test_hourly_cuts_remain_lower_bounds_with_ramps_and_multiple_scenarios(tiny):
    d, _ = tiny()
    d.ramp_up[0], d.cost_b[1], d.cost_a[1] = 15., 5., .02
    scenarios = [NS(name="low", probability=.4, load=np.array([30., 80., 35.])),
                 NS(name="high", probability=.6, load=np.array([35., 85., 40.]))]
    m = model(d, scenarios)
    full = partition(m, 3)[0][0]
    hourly, feasible = HourlyBounds(full), []
    for bits in product([0., 1.], repeat=6):
        x = pack(d, np.asarray(bits).reshape((2, 3), order="F"))
        r = dispatch(full, x, 5, certified=True)
        if r.feasible:
            actual = hourly.recourse(r.point)
            assert actual.sum() == pytest.approx(r.recourse)
            list(hourly.observe(x, "test", actual, lambda: 5.))
            feasible.append((x, r, actual))
    assert feasible and hourly.cuts
    relaxed = []
    for x, r, actual in feasible:
        for cut in hourly.cuts:
            assert cut["alpha"] + np.asarray(cut["beta"]) @ x[cut["binary_indices"]] <= actual[cut["hour"]-1] + 1e-7
        bound = sum(max(c["alpha"] + np.asarray(c["beta"]) @ x[c["binary_indices"]]
                        for c in hourly.cuts if c["hour"] == hour+1) for hour in range(d.horizon))
        relaxed.append(r.recourse-bound)
    assert max(relaxed) > 1., "Inter-hour ramping is relaxed only in the auxiliary hourly bounds"
    assert min(r.cost for _, r, _ in feasible) == pytest.approx(optimize(m, 5, integral=True).objval)


def test_bound_probes_keep_infeasible_points_and_do_not_repair(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    cert = whole.certifier
    old_cost = whole.result.objective
    monkeypatch.setattr(solver, "repair", lambda *args: pytest.fail("bound probe must not be repaired"))
    row = cert.probe_advice(np.zeros((2, 3)), {"proposal_id": "probe:1"})
    assert not row["feasible"] and row["cut_added"] and row["outcome"] == "infeasible"
    assert whole.result.objective == old_cost
    cut_count = len(cert.cuts)
    again = cert.probe_advice(np.zeros((2, 3)), {"proposal_id": "probe:2"})
    assert again["duplicate"] and not again["cut_added"] and len(cert.cuts) == cut_count
    assert whole.diagnostics["certificate_cuts"][-1]["source"] == "llm_bound"


@pytest.mark.parametrize("raw", [np.full((2, 3), .5), np.full((2, 3), np.nan),
                                 np.zeros((1, 3)), np.array([[1, 1, 0], [1, 1, 1]])])
def test_invalid_bound_probes_never_reach_dispatch(tiny, options, monkeypatch, raw):
    whole = whole_day(tiny, options)
    monkeypatch.setattr(solver, "dispatch", lambda *a, **k: pytest.fail("invalid probe dispatched"))
    row = whole.certifier.probe_advice(raw, {"proposal_id": "bad"})
    assert row["outcome"] in ("invalid_matrix", "discrete_infeasible")
    assert not row["cut_added"]


def fake_master(cert, monkeypatch, lower=220., optimize_fn=lambda: None):
    cert.g = NS(Params=NS(), Status=9, ObjBound=lower, ObjVal=-999., SolCount=0, optimize=optimize_fn)
    monkeypatch.setattr(cert, "_sync", lambda: None)


def test_time_limited_master_uses_bound_without_an_incumbent(tiny, options, monkeypatch):
    whole = whole_day(tiny, options, n=1)
    fake_master(whole.certifier, monkeypatch)
    whole.certifier.step()
    assert whole.result.lower_bound == pytest.approx(220.-1e-5)
    row = whole.result.certification_history[-1]
    assert row["status"] == "TIME_LIMIT" and row["bound_used"]
    assert "probe_hash" not in row


def test_late_master_bound_is_not_used(tiny, options, monkeypatch):
    whole = whole_day(tiny, options, n=1)
    now = [0.]
    whole.deadline = 10.
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    monkeypatch.setattr(certification, "perf_counter", lambda: now[0])
    fake_master(whole.certifier, monkeypatch, optimize_fn=lambda: now.__setitem__(0, 11.))
    with pytest.raises(TimeoutError):
        whole.certifier.step()
    assert whole.result.lower_bound == 0.
    assert not whole.result.certification_history[-1]["bound_used"]


def test_crossed_bound_is_an_error_and_not_clamped(tiny, options):
    whole = whole_day(tiny, options, n=1)
    with pytest.raises(RuntimeError, match="exceeds"):
        whole.certifier.update_bound(whole.result.objective+.01)
    assert whole.result.lower_bound == 0.


def test_master_and_global_cuts_survive_repricing(tiny, options):
    whole = whole_day(tiny, options)
    whole.o.bound_slice_seconds = .2
    cert = whole.certifier
    try:
        cert.step()
        first_model, first_cut = cert.g, cert.cuts[0]
        partition(whole.m, 1, np.full(len(whole.m.rhs), .5))
        cert.step()
        assert cert.g is first_model and cert.cuts[0] is first_cut
        assert cert.b is whole.full and cert.b.constant == 0.
    finally:
        cert.close()


@pytest.mark.parametrize("change", [{"bound_mode": "unknown"}, {"bound_slice_seconds": 0},
                                    {"bound_slice_seconds": float("nan")}, {"bound_slice_seconds": True}])
def test_certification_option_validation(change, tmp_path):
    with pytest.raises(ValueError):
        load_options(change, tmp_path / "missing.json")
