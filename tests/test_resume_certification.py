from dataclasses import asdict

import numpy as np
import pytest

from run_suc import _serializable
from suc_benders import certification, solver
from test_certification import whole_day


def saved_state(tiny, options):
    whole = whole_day(tiny, options)
    options.bound_slice_seconds = .2
    try:
        whole.certifier.step()
        state = _serializable(asdict(whole.result))
    finally:
        whole.certifier.close()
    state.update(charged_elapsed_seconds=2., resume_cursor=dict(outer=1, round=2),
                 resume=dict(source="test", block_state_rebuilt=True), llm_calls=3,
                 llm_history=[dict(status="OK", background_seconds=.1) for _ in range(3)])
    state["lower_bound"] = min(state["lower_bound"], 220.)
    for row in state["certification_history"]:
        row["lower"] = min(row["lower"], 220.)
    state["diagnostics"].update(sa_calls=8, complete_rounds=1, coordination_count=0)
    return state


def test_resume_rebuilds_blocks_and_preserves_global_cuts_and_llm_limit(tiny, options, monkeypatch):
    state = saved_state(tiny, options)
    d, scenarios = tiny()
    options.llm_enabled = True
    options.max_outer, options.max_inner = 1, 2
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    monkeypatch.setattr(certification.bound_advisor, "BoundAdvisor",
                        lambda *a, **k: pytest.fail("Three existing LLM calls exhaust the request budget"))
    sync = certification.Certification._sync
    synchronized = []

    def checked_sync(cert):
        sync(cert)
        cert.g.update()
        synchronized.append((cert.synced, cert.hourly_synced, cert.g.NumConstrs))

    monkeypatch.setattr(certification.Certification, "_sync", checked_sync)
    result = solver.solve(d, scenarios, options, resume_state=state)
    assert result.status == "CONVERGED", result.message
    assert result.objective == pytest.approx(225.)
    assert state["lower_bound"] <= result.lower_bound <= result.objective+1e-5
    assert result.elapsed_seconds >= 2.
    assert result.diagnostics["timings"]["prior_execution"] == 2.
    assert result.llm_calls == 3 and len(result.llm_history) == 3
    assert result.diagnostics["rounds"][0]["round"] == 2
    assert result.diagnostics["complete_rounds"] == 2
    for name in ("certificate_cuts", "hourly_certificate_cuts"):
        old = [cut["cut_id"] for cut in state["diagnostics"][name]]
        actual = [cut["cut_id"] for cut in result.diagnostics[name]]
        assert actual[:len(old)] == old
    assert synchronized and synchronized[0][2] > synchronized[0][0]+synchronized[0][1]
    assert synchronized[0][1] >= len(state["diagnostics"]["hourly_certificate_cuts"])
    assert len(result.certification_history) > len(state["certification_history"])
    assert all("u" in row for row in result.diagnostics["evaluations"])


@pytest.mark.parametrize("corruption", ["bound", "incumbent"])
def test_resume_rejects_invalid_recovered_evidence(tiny, options, corruption):
    state = saved_state(tiny, options)
    d, scenarios = tiny()
    if corruption == "bound":
        state["lower_bound"] = state["objective"]+1.
    else:
        state["u"] = np.zeros((d.n_units, d.horizon)).tolist()
    result = solver.solve(d, scenarios, options, resume_state=state)
    assert result.status == "RESUME_FAILED"


@pytest.mark.parametrize("prior", [-1., float("nan"), 30., 31.])
def test_resume_cannot_reset_or_extend_exhausted_budget(tiny, options, prior):
    d, scenarios = tiny()
    with pytest.raises(ValueError, match="resume budget"):
        solver.solve(d, scenarios, options, resume_state={"charged_elapsed_seconds": prior})


@pytest.mark.parametrize("change", [None, "cold_cost", "cold_hours", "initial_history", "scenario", "missing"])
def test_cold_resume_fingerprint(tiny, options, monkeypatch, change):
    def cold_tiny(n=2):
        d, scenarios = tiny(n)
        d.cold_startup_cost = np.full(n, 11.)
        d.cold_start_hours = np.ones(n)
        return d, scenarios

    state = saved_state(cold_tiny, options)
    d, scenarios = cold_tiny()
    options.max_outer, options.max_inner = 1, 2
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    if change == "cold_cost":
        d.cold_startup_cost += 1
    elif change == "cold_hours":
        d.cold_start_hours += 1
    elif change == "initial_history":
        d.initial_duration += 1
    elif change == "scenario":
        scenarios[0].name = "renamed"
    elif change == "missing":
        del state["diagnostics"]["model_fingerprint"]
    if change:
        # All of these leave the saved incumbent's hot startup fees unchanged.
        # Reject before any old cut can enter the new master.
        monkeypatch.setattr(certification.Certification, "restore_cuts",
                            lambda *args: pytest.fail("Mismatched cuts must not be loaded"))
    result = solver.solve(d, scenarios, options, resume_state=state)
    if change:
        assert result.status == "RESUME_FAILED" and "fingerprint" in result.message
    else:
        assert result.status == "CONVERGED", result.message
        assert result.validation["feasible"] and result.lower_bound <= result.objective+1e-5
        assert result.diagnostics["model_fingerprint"] == state["diagnostics"]["model_fingerprint"]


def test_legacy_fixed_resume_without_fingerprint(tiny, options, monkeypatch):
    state = saved_state(tiny, options)
    del state["diagnostics"]["model_fingerprint"]
    for row in state["diagnostics"]["evaluations"]:
        if "hourly_recourse_costs" in row:
            row["hourly_fuel_costs"] = row.pop("hourly_recourse_costs")
    for row in state["diagnostics"]["bound_probes"]:
        if "hourly_actual_recourse" in row:
            row["hourly_actual_fuel"] = row.pop("hourly_actual_recourse")
    d, scenarios = tiny()
    options.max_outer, options.max_inner = 1, 2
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    result = solver.solve(d, scenarios, options, resume_state=state)
    assert result.status == "CONVERGED", result.message
    assert result.objective == pytest.approx(225.)
    assert len(result.diagnostics["model_fingerprint"]) == 64
    assert all("hourly_fuel_costs" not in row for row in result.diagnostics["evaluations"])
    assert all("hourly_actual_fuel" not in row for row in result.diagnostics["bound_probes"])


def test_legacy_resume_rejects_different_dimensions(tiny, options):
    state = saved_state(tiny, options)
    del state["diagnostics"]["model_fingerprint"]
    d, scenarios = tiny(1)
    result = solver.solve(d, scenarios, options, resume_state=state)
    assert result.status == "RESUME_FAILED" and "dimensions" in result.message
