from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import dual_advisor, solver
from suc_benders.validation import pack
from test_certification import whole_day


@pytest.fixture
def scheduled(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    options.llm_enabled, options.llm_max_calls = True, None
    cert = whole.certifier
    cert.last_master = dict(x=pack(whole.d, whole.result.u), predicted_cost=100.)
    monkeypatch.setattr(cert, "converged", lambda: False)
    monkeypatch.setattr(cert, "step", lambda: None)
    calls = []

    def launch(d, o, body, iteration, seconds):
        calls.append(json.loads(body["input"]))
        return NS(poll=lambda: ([], dict(status="FAILED", proposals=[], seconds=0.)))

    monkeypatch.setattr(dual_advisor.bound_advisor, "BoundAdvisor", launch)
    return whole, calls


def complete(whole, number, upper_drop=0., lower_gain=0., sampled=True, allow=True):
    upper, lower, sa = whole.result.objective, whole.result.lower_bound, whole.diagnostics["sa_calls"]
    whole.result.objective -= upper_drop
    whole.result.lower_bound += lower_gain
    whole.diagnostics["sa_calls"] += int(sampled)
    whole.advice.after_round(upper, lower, sa, [], 1, number, allow)


def test_three_rounds_alternate_three_each_then_return_to_sa(scheduled):
    whole, calls = scheduled
    for i in range(1, 3):
        complete(whole, i)
    assert not calls
    complete(whole, 3)
    assert [p["advice_routing"]["channel"] for p in calls] == ["upper", "lower"]*3
    assert [p["advice_routing"]["batch_call"] for p in calls] == [1, 1, 2, 2, 3, 3]
    assert whole.advice.state["phase"] == "sa"
    assert whole.advice.state["stalled"] == dict(upper=0, lower=0)
    for i in range(4, 6):
        complete(whole, i)
    assert len(calls) == 6
    complete(whole, 6)
    assert len(calls) == whole.result.llm_calls == 12
    assert [b["trigger_sa_round"] for b in whole.diagnostics["llm_batches"]] == [3, 6]


@pytest.mark.parametrize("improving,requested", [("upper", "lower"), ("lower", "upper")])
def test_stagnation_is_separate_for_each_bound(scheduled, improving, requested):
    whole, calls = scheduled
    for i in range(3):
        complete(whole, i, upper_drop=1. if improving == "upper" else 0.,
                 lower_gain=1. if improving == "lower" else 0.)
    assert [p["advice_routing"]["channel"] for p in calls] == [requested]*3


@pytest.mark.parametrize("maximum,expected", [(0, 0), (1, 1), (3, 3), (8, 8), (None, 12)])
def test_global_finite_zero_and_unlimited_caps(scheduled, maximum, expected):
    whole, calls = scheduled
    whole.o.llm_max_calls = maximum
    for i in range(6):
        complete(whole, i)
    assert len(calls) == whole.result.llm_calls == expected


def test_only_sampled_complete_rounds_trigger_and_tail_is_disabled(scheduled):
    whole, calls = scheduled
    for i in range(5):
        whole.certifier.step()
        complete(whole, i, sampled=False)
    assert not calls and whole.advice.state["sa_rounds"] == 0
    for i in range(3):
        complete(whole, i, allow=False)
    assert not calls


def test_significant_gain_ends_only_that_channel_and_refreshes_context(scheduled, monkeypatch):
    whole, calls = scheduled
    def launch(d, o, body, iteration, seconds):
        prompt = json.loads(body["input"])
        calls.append(prompt)
        def poll():
            if iteration == 1:
                whole.result.objective -= 2.
            return [], dict(status="FAILED", proposals=[], seconds=0.)
        return NS(poll=poll)
    monkeypatch.setattr(dual_advisor.bound_advisor, "BoundAdvisor", launch)
    for i in range(3):
        complete(whole, i)
    assert [p["advice_routing"]["channel"] for p in calls] == ["upper", "lower", "lower", "lower"]
    assert calls[1]["context"]["bound_search"]["upper_bound"] == whole.result.objective


def test_pending_request_allows_certification_without_second_request(scheduled, monkeypatch):
    whole, calls = scheduled
    cycles = []
    monkeypatch.setattr(whole.certifier, "step", lambda: cycles.append(1))
    def launch(d, o, body, iteration, seconds):
        calls.append(iteration)
        polls = []
        def poll():
            polls.append(1)
            if len(polls) < 3:
                assert len(calls) == iteration
                return None
            return [], dict(status="FAILED", proposals=[], seconds=0.)
        return NS(poll=poll)
    monkeypatch.setattr(dual_advisor.bound_advisor, "BoundAdvisor", launch)
    for i in range(3):
        complete(whole, i)
    assert calls == list(range(1, 7)) and len(cycles) == 18
    assert whole.diagnostics["sa_calls"] == 3


def test_upper_repairs_and_lower_does_not_and_both_can_add_cuts(tiny, options, monkeypatch):
    for channel in ("upper", "lower"):
        whole = whole_day(tiny, options)
        a, cert = whole.advice, whole.certifier
        raw = np.zeros((2, 3)) if channel == "upper" else np.array([[1, 1, 1], [0, 0, 0]])
        record = {"proposal_id": f"{channel}:1"}
        a.begin([channel])
        a.pending = dict(channel=channel)
        a.baseline = whole.upper_context()[2]
        a.worker = NS(poll=lambda: ([(raw, record)], dict(status="OK", proposals=[record], seconds=0.)))
        before, cuts = whole.result.objective, len(cert.cuts)
        if channel == "lower":
            monkeypatch.setattr(whole, "prepare", lambda *a: pytest.fail("lower probe must not be repaired"))
        assert a.poll([])
        assert whole.result.objective < before and len(cert.cuts) > cuts
        assert a.pending["direct_incumbent_decrease"] == pytest.approx(before-whole.result.objective)
        assert a.pending["numerical_evaluation_completed"]
        if channel == "upper":
            assert record["hamming_u"] > 0 and record["repaired_discrete_feasible"]
            assert "feedback" in a.pending
        assert cert.cuts[-1].certified
        cert.close()


def test_late_reply_never_changes_bounds_or_cuts(scheduled, monkeypatch):
    whole, _ = scheduled
    a = whole.advice
    a.begin(["lower"])
    record = {"proposal_id": "late"}
    a.pending = dict(channel="lower")
    a.worker = NS(poll=lambda: ([(np.zeros((2, 3)), record)], dict(status="OK", proposals=[record])))
    before = len(whole.certifier.cuts), whole.result.objective, whole.result.lower_bound
    monkeypatch.setattr(whole, "remaining", lambda: -1.)
    a.poll([])
    assert a.pending["status"] == "TIME_LIMIT"
    assert before == (len(whole.certifier.cuts), whole.result.objective, whole.result.lower_bound)


def test_restore_keeps_partial_batch_and_spent_calls(scheduled):
    whole, calls = scheduled
    a = whole.advice
    a.state["stalled"] = dict(upper=3, lower=3)
    a.state["sa_rounds"] = 3
    a.begin(["upper", "lower"])
    a.state["batch"]["counts"] = dict(upper=2, lower=1)
    a.state["batch"]["last_channel"] = "upper"
    whole.result.llm_calls = 3
    whole.result.llm_history.append(dict(status="RUNNING"))
    saved = deepcopy(whole.diagnostics)
    a.restore(saved)
    assert whole.result.llm_history[-1]["status"] == "CANCELLED"
    a.run_batch([])
    assert [p["advice_routing"]["channel"] for p in calls] == ["lower", "upper", "lower"]
    assert whole.result.llm_calls == 6
    assert whole.diagnostics["llm_batches"][-1]["counts"] == dict(upper=3, lower=3)


def test_legacy_restore_starts_new_sa_window(scheduled):
    whole, _ = scheduled
    whole.advice.restore({})
    assert whole.advice.state["phase"] == "sa" and whole.advice.state["stalled"] == dict(upper=0, lower=0)
    assert "migration" in whole.advice.state


def test_restored_shared_receipt_list_records_each_request_once(scheduled):
    whole, calls = scheduled
    whole.diagnostics["llm_calls"] = whole.result.llm_history
    for i in range(3):
        complete(whole, i)
    assert len(calls) == whole.result.llm_calls == len(whole.result.llm_history) == 6
    assert len({r["call_id"] for r in whole.result.llm_history}) == 6


def test_solver_round_wiring_returns_to_sa_between_bursts(tiny, options, monkeypatch):
    from suc_benders.certification import Certification
    from test_search import force_open_master
    d, scenarios = tiny(2, 6)
    options.bound_mode, options.llm_enabled, options.llm_max_calls = "integer_benders", True, None
    options.max_outer, options.max_inner = 1, 7
    force_open_master(monkeypatch)
    events = []
    monkeypatch.setattr(solver, "anneal", lambda *a: (events.append("sa") or [], {"bits": 1}))
    def step(cert):
        cert.last_master = dict(x=pack(d, cert.result.u), predicted_cost=100.)
    monkeypatch.setattr(Certification, "step", step)
    monkeypatch.setattr(Certification, "converged", lambda cert: cert.whole.diagnostics["sa_calls"] >= 14)
    def launch(d, o, body, iteration, seconds):
        events.append(json.loads(body["input"])["advice_routing"]["channel"])
        return NS(poll=lambda: ([], dict(status="FAILED", proposals=[], seconds=0.)))
    monkeypatch.setattr(dual_advisor.bound_advisor, "BoundAdvisor", launch)
    result = solver.solve(d, scenarios, options)
    assert result.status == "CONVERGED", result.message
    assert result.llm_calls == 12 and result.diagnostics["complete_rounds"] == 7
    assert events == ["sa"]*6+["upper", "lower"]*3+["sa"]*6+["upper", "lower"]*3+["sa"]*2


def test_validation_batch_stops_after_each_channels_first_numeric_reply(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    a, cert = whole.advice, whole.certifier
    options.llm_enabled, options.llm_max_calls = True, None
    cert.last_master = dict(x=pack(whole.d, whole.result.u), predicted_cost=100.)
    monkeypatch.setattr(cert, "converged", lambda: True)
    monkeypatch.setattr(cert, "step", lambda: None)
    calls = []
    def launch(d, o, body, iteration, seconds):
        channel = json.loads(body["input"])["advice_routing"]["channel"]
        calls.append(channel)
        raw = np.array([[1, 1, 1], [0, 0, 0]]) if channel == "upper" else np.zeros((2, 3))
        row = dict(proposal_id=f"validation:{iteration}")
        return NS(poll=lambda: ([(raw, row)], dict(status="OK", proposals=[row], seconds=0.)))
    monkeypatch.setattr(dual_advisor.bound_advisor, "BoundAdvisor", launch)
    a.begin(["upper", "lower"], validation=True)
    a.run_batch([])
    assert calls == ["upper", "lower"]
    assert all(r["numerical_evaluation_completed"] and r["phase"] == "validation" for r in whole.result.llm_history)
    assert whole.diagnostics["llm_batches"][-1]["done"] == dict(upper=True, lower=True)
