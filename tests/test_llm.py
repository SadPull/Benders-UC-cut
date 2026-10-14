import io
import json
from time import perf_counter

import numpy as np
import pytest

from suc_benders import llm, solver
from suc_benders.config import load_options
from suc_benders.model import model, partition
from suc_benders.validation import pack


def response(rows):
    return {"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"candidates": rows})}]}]}


@pytest.fixture
def whole(tiny, options):
    d, s = tiny(3, 5)
    m = model(d, s); full = partition(m, 5)[0][0]; started = perf_counter()
    w = solver.WholeDay(m, full, options, started+30, solver.Result(), started)
    assert w.accept(pack(d, np.ones((3, 5))), "initial")
    options.llm_url, options.llm_api_key, options.llm_model = "https://example.invalid/v1/responses", "offline-secret", "offline-model"
    return w


def test_whole_day_prompt_and_http_fields(whole):
    body = llm.request_body(whole.d, whole.scenarios, whole.result, whole.o, {"marker": 1}, [whole.result], 1)
    assert body["stream"] is False and body["store"] is False
    assert body["reasoning"] == {"effort": "max"}
    assert "offline-secret" not in json.dumps(body)
    data = json.loads(body["input"])
    assert data["horizon"] == 5 and data["units"] == 3 and data["requested_candidates"] == 4
    assert data["context"] == {"marker": 1}
    assert "unit_hour_decommit_gradient" in body["instructions"]


def test_parser_accepts_four_whole_day_u_proposals(whole):
    raw = whole.result.u.tolist()
    rows, info = llm.parse_response(response([{"u": raw}]*5), whole.d)
    assert len(rows) == info["format_valid"] == 4
    np.testing.assert_array_equal(rows[0][0], whole.result.u)
    # Duplicates are resolved after repair, not before rebuilding v/w.
    assert len(info["proposals"]) == 4


@pytest.mark.parametrize("bad", [None, {}, {"status": "incomplete"}, {"status": "completed", "output": []},
                                {"status": "completed", "output": [{"type": "message", "content": [
                                    {"type": "output_text", "text": "invalid"}]}]}])
def test_bad_responses(whole, bad):
    candidates, info = llm.parse_response(bad, whole.d)
    assert not candidates and info["status"] == "FAILED"


@pytest.mark.parametrize("value", [[[1]], "not a matrix", None])
def test_bad_matrix(whole, value):
    candidates, info = llm.parse_response(response([{"u": value}]), whole.d)
    assert not candidates and info["format_valid"] == 0


def test_transport_and_timeout(whole, monkeypatch):
    seen = {}

    def post(req, timeout):
        seen.update(body=json.loads(req.data), timeout=timeout, url=req.full_url,
                    method=req.get_method(), headers=req.headers)
        return io.BytesIO(json.dumps(response([{"u": whole.result.u.tolist()}])).encode())

    monkeypatch.setattr(llm, "urlopen", post)
    args = (whole.d, whole.scenarios, whole.result, whole.o, 10, {}, [], 1)
    candidates, info = llm.request(*args)
    assert candidates and info["status"] == "OK" and 0 < seen["timeout"] <= 10
    assert seen["url"] == whole.o.llm_url and seen["method"] == "POST"
    assert seen["headers"]["Authorization"] == "Bearer offline-secret"
    assert seen["headers"]["Content-type"] == "application/json"

    def failed(*args, **kwargs):
        raise RuntimeError("offline-secret must not appear in a result")

    monkeypatch.setattr(llm, "urlopen", failed)
    candidates, info = llm.request(*args)
    assert not candidates and info["status"] == "FAILED" and "offline-secret" not in str(info)
    monkeypatch.setattr(llm, "urlopen", post)
    times = iter([0., 0., 11., 11.])
    monkeypatch.setattr(llm, "perf_counter", lambda: next(times))
    candidates, info = llm.request(*args)
    assert not candidates and info["status"] == "TIME_LIMIT"


def test_round_advice_refreshes_full_incumbent_feedback_and_all_projections(whole, monkeypatch):
    whole.o.llm_enabled = True; whole.o.llm_patience = 1
    searches = [solver.BlockSearch(b, whole.o, 1, j+1, whole)
                for j, b in enumerate(partition(whole.m, 3)[0])]
    seen = []

    def request(d, scenarios, best, o, seconds, context, bases, iteration):
        seen.append((best.u.copy(), context))
        raw = best.u.copy()
        if iteration <= 2:
            raw[iteration-1] = 0
        record = {"proposal_id": f"llm:{iteration}:0"}
        return [(raw, record)], {"status": "OK", "message": "", "proposals": [record]}

    monkeypatch.setattr(llm, "request", request)
    for inner in range(1, 4):
        whole.after_round(whole.result.objective, searches, 1, inner, True)
        assert whole.result.llm_calls == inner
    assert whole.result.llm_calls == 3
    assert seen[0][0].all() and not seen[1][0][0].any() and not seen[2][0][:2].any()
    assert seen[1][1]["previous_call_feedback"]["proposals"][0]["cost_change"]["total"] < 0
    context = seen[0][1]
    assert context["scenario_duals"]["balance"] and context["unit_hour_decommit_gradient"]
    assert context["incumbent_cost"]["total"] > whole.result.objective
    for call in whole.result.llm_history:
        projections = call["proposals"][0]["block_projections"]
        assert [x["block"] for x in projections] == [1, 2]
        assert all(x["completed"] and x["feasible"] for x in projections)
    assert all(search.cuts and search.planes for search in searches)
    assert len(whole.advisor_pool) == 2
    assert whole.result.llm_history[-1]["proposals"][0]["outcome"] == "duplicate"
    assert not whole.result.u[:2].any() and whole.result.validation["feasible"]


@pytest.mark.parametrize("max_calls", [3, None])
def test_late_llm_output_is_not_evaluated(whole, monkeypatch, max_calls):
    whole.o.llm_enabled = True; whole.o.llm_patience = 1
    whole.o.llm_max_calls = max_calls
    before = whole.result.objective; now = [0.]; whole.deadline = 10
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    b = partition(whole.m, 3)[0][0]

    def late(*args):
        now[0] = 11.
        return [(np.zeros((3, 5)), {})], {"status": "OK", "proposals": []}

    monkeypatch.setattr(llm, "request", late)
    search = solver.BlockSearch(b, whole.o, 1, 1, whole)
    monkeypatch.setattr(search, "evaluate", lambda *args: pytest.fail("late projection"))
    whole.after_round(before, [search], 1, 1, True)
    assert whole.result.llm_calls == 1 and whole.result.objective == before and not whole.advisor_pool


def test_scenario_duals_are_unweighted_once(whole):
    from types import SimpleNamespace as NS
    d = whole.d
    scenarios = [NS(name="a", probability=.25, load=whole.scenarios[0].load.copy()),
                 NS(name="b", probability=.75, load=whole.scenarios[0].load.copy())]
    m = model(d, scenarios); full = partition(m, d.horizon)[0][0]; start = perf_counter()
    joint = solver.WholeDay(m, full, whole.o, start+30, solver.Result(), start)
    assert joint.accept(pack(d, whole.result.u), "initial")
    for dispatch in joint.dispatches:
        np.testing.assert_allclose(dispatch.balance_dual, whole.dispatches[0].balance_dual, atol=2e-5)
        np.testing.assert_allclose(dispatch.reserve_dual, whole.dispatches[0].reserve_dual, atol=2e-5)


def test_local_configuration_and_removed_option(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"kaiwu": {"user_id": "test", "sdk_code": "test-secret"},
                               "llm": {"enabled": True, "url": "https://example.invalid", "api_key": "test-key", "model": "test"}}))
    o = load_options({"llm_enabled": False}, path)
    assert not o.llm_enabled and o.kaiwu_sdk_code == "test-secret"
    assert "test-secret" not in repr(o) and "test-key" not in repr(o)
    with pytest.raises(ValueError):
        load_options({"llm_api_key": ""}, path)
    with pytest.raises(ValueError, match="certify_seconds"):
        load_options({"certify_seconds": 5}, path)
