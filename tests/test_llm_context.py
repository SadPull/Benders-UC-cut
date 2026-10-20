import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from conftest import whole_day
from suc_benders import llm
from suc_benders.context import LLMContextInput, build_llm_context
from suc_benders.validation import pack


def test_context_drops_duplicates_and_averages_marginal_cost_over_on_hours(tiny, options):
    whole = whole_day(tiny, options, n=3)
    u = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=float)
    assert whole.accept(pack(whole.d, u), "test")
    context = build_llm_context(whole.d, whole.scenarios,
                                LLMContextInput(whole.result, whole.dispatches, [], []))
    for key in ("top_decommit_opportunities", "tight_constraints", "base_candidates"):
        assert key not in context
    p = sum(s.probability * np.asarray(r.p, float) for s, r in zip(whole.scenarios, whole.dispatches))
    rows = context["unit_marginal_cost"]
    expected = 2 * whole.d.cost_a[1] * p[1, 0] + whole.d.cost_b[1]
    assert np.isclose(rows[1]["at_current_p"], round(expected, 3))
    assert rows[2]["at_current_p"] is None
    _, user = llm._prompt(whole.d, whole.scenarios, 1, whole.result, [], 4, context)
    assert json.loads(user)["context"]["unit_marginal_cost"][2]["at_current_p"] is None


def test_continuation_prompt_sends_only_dynamic_snapshot(tiny, options):
    whole = whole_day(tiny, options, n=3)
    context = build_llm_context(whole.d, whole.scenarios,
                                LLMContextInput(whole.result, whole.dispatches, [], []))
    fixed = np.full((3, whole.d.horizon), -1); fixed[0] = 1
    context['reachability'] = dict(fixed_u=fixed.tolist())
    system, full = llm._prompt(whole.d, whole.scenarios, 1, whole.result, [], 4, context)
    assert llm._prompt(whole.d, whole.scenarios, 1, whole.result, [], 4, context, False) == (system, full)
    delta_system, delta = llm._prompt(whole.d, whole.scenarios, 2, whole.result, [], 4, context, True)
    full, delta = json.loads(full), json.loads(delta)
    assert system and delta_system == ""
    assert set(delta) == set(llm.DYNAMIC_KEYS) | {"static_inputs"}
    for key in ("physical_rules", "rules", "indexing", "scenarios", "pmin", "cost_a", "output_schema"):
        assert key in full and key not in delta
    assert "reachability" in full["context"] and "reachability" not in delta["context"]
    # 梯度仍按 fixed_u 过滤，与完整提示一致。
    assert delta["context"]["unit_hour_decommit_gradient"] == full["context"]["unit_hour_decommit_gradient"]
    assert delta["context"]["decommit_filter"] == full["context"]["decommit_filter"]
    assert all(row["unit"] != 0 for row in delta["context"]["unit_hour_decommit_gradient"])
    assert delta["incumbent_u"] == full["incumbent_u"] and delta["iteration"] == 2


class _Stream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args):
        return False


def test_post_response_reads_final_event_from_stream(monkeypatch):
    final = {"status": "completed", "usage": {"output_tokens": 3}, "output": []}
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(body=json.loads(request.data), accept=request.get_header("Accept"), timeout=timeout)
        return _Stream([b"event: response.created\n", b'data: {"type":"response.created"}\n', b"\n",
                        b'data: {"type":"response.reasoning_summary_text.delta","delta":"x"}\n',
                        b"data: " + json.dumps({"type": "response.completed", "response": final}).encode() + b"\n"])

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    assert llm.post_response("http://x/v1/responses", "k", {"model": "m", "stream": False}, 7) == final
    assert sent == dict(body={"model": "m", "stream": True}, accept="text/event-stream", timeout=7)


@pytest.mark.parametrize("lines", [[b'data: {"type":"response.created"}\n', b"data: [DONE]\n"],
                                   [b'data: {"type":"error","message":"overloaded"}\n']])
def test_post_response_rejects_stream_without_final_response(monkeypatch, lines):
    monkeypatch.setattr(llm, "urlopen", lambda request, timeout: _Stream(lines))
    with pytest.raises(RuntimeError):
        llm.post_response("http://x/v1/responses", "k", {"model": "m"}, 7)


def _reply(candidates):
    text = json.dumps({"candidates": candidates})
    return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def _candidate(*changes):
    return {"name": "c", "reason": "r", "changes": [
        {"unit": unit, "on_intervals": [{"start": s, "end": e} for s, e in intervals]} for unit, intervals in changes]}


SHAPE = NS(n_units=3, horizon=4)
BASE = [[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]


def test_changes_replace_listed_rows_and_keep_the_rest():
    candidates, info = llm.parse_response(_reply([_candidate((2, [(1, 2)]), (3, []))]), SHAPE, 5, BASE)
    assert info["status"] == "OK" and info["message"] == "" and info["invalid"] == 0
    u, proposal = candidates[0]
    assert u.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]
    assert proposal["raw_u"] == u.tolist() and proposal["raw_changes"][0]["unit"] == 2
    assert proposal["proposal_id"] == "llm:5:0"


@pytest.mark.parametrize("bad,reason", [
    (_candidate((4, [(1, 1)])), "unit 4 out of range 1..3"),
    (_candidate((1, [(3, 2)])), "interval 3-2 of unit 1 invalid for hours 1..4"),
    (_candidate((1, [(1, 5)])), "interval 1-5 of unit 1 invalid for hours 1..4"),
    (_candidate((1, [(1, 2)]), (1, [(3, 4)])), "duplicate unit 1"),
    (_candidate((1, [(1, 3), (3, 4)])), "overlapping intervals for unit 1"),
    ({"name": "c", "reason": "r", "changes": []}, "changes must be a non-empty list"),
    ({"name": "c", "reason": "r", "u": [[1]]}, "malformed candidate"),
])
def test_invalid_candidate_is_dropped_and_recorded_while_others_survive(bad, reason):
    good = _candidate((2, [(3, 4)]))
    candidates, info = llm.parse_response(_reply([good, bad, good]), SHAPE, 1, BASE)
    assert info["status"] == "OK" and info["accepted"] == 2 and info["format_valid"] == 2
    assert info["invalid"] == 1 and info["message"] == "1 of 3 candidates invalid"
    assert [p["proposal_id"] for _, p in candidates] == ["llm:1:0", "llm:1:2"]
    assert info["proposals"][1]["outcome"] == "invalid_candidate"
    assert info["proposals"][1]["invalid_reason"] == reason and "raw_u" not in info["proposals"][1]


def test_all_invalid_candidates_fail_but_stay_recorded():
    candidates, info = llm.parse_response(_reply([_candidate((9, []))] * 2), SHAPE, 1, BASE)
    assert candidates == [] and info["status"] == "FAILED" and info["message"] == "NO_VALID_CANDIDATE"
    assert info["invalid"] == 2 and all(p["outcome"] == "invalid_candidate" for p in info["proposals"])


def test_missing_base_and_broken_json_are_reported():
    candidates, info = llm.parse_response(_reply([_candidate((1, []))]), SHAPE, 1, None)
    assert candidates == [] and info["proposals"][0]["invalid_reason"] == "missing base schedule"
    broken = {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": '{"candidates": ['}]}]}
    assert llm.parse_response(broken, SHAPE, 1, BASE)[1]["message"] == "INVALID_RESPONSE"
    # 模型主动不给候选不算失败。
    assert llm.parse_response(_reply([]), SHAPE, 1, BASE)[1]["status"] == "OK"


def test_both_transports_share_the_output_schema(tiny, options, tmp_path, monkeypatch):
    whole = whole_day(tiny, options, n=3)
    body = llm.request_body(whole.d, whole.scenarios, whole.result, options, None, [], 1)
    schema = llm.output_schema(whole.d.n_units, whole.d.horizon)
    assert body["text"]["format"] == {"type": "json_schema", "name": "uc_candidates", "strict": True, "schema": schema}
    change = schema["properties"]["candidates"]["items"]["properties"]["changes"]["items"]
    assert change["properties"]["unit"]["maximum"] == whole.d.n_units
    assert change["properties"]["on_intervals"]["items"]["properties"]["end"]["maximum"] == whole.d.horizon
    assert llm.prompt_base(body) == whole.result.u.tolist()
    assert llm.prompt_base({"input": "not json"}) is None
    from suc_benders import advisor_request
    monkeypatch.setattr(advisor_request.shutil, "which", lambda command: command)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    advisor_request._prepare_codex("codex", body, NS(n_units=whole.d.n_units, horizon=whole.d.horizon), tmp_path)
    assert json.loads((tmp_path/"schema.json").read_text(encoding="utf-8")) == schema
