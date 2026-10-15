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
    rows, info = llm.parse_response(response([{"u": raw}]*4), whole.d)
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
    body = llm.request_body(whole.d, whole.scenarios, whole.result, whole.o, {}, [], 1)
    raw = llm.post_response(whole.o.llm_url, whole.o.llm_api_key, body, 10.)
    candidates, info = llm.parse_response(raw, whole.d)
    assert candidates and info["status"] == "OK" and seen["timeout"] == 10.
    assert seen["url"] == whole.o.llm_url and seen["method"] == "POST"
    assert seen["headers"]["Authorization"] == "Bearer offline-secret"
    assert seen["headers"]["Content-type"] == "application/json"


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


@pytest.mark.parametrize('value', [np.full((3, 5), .5).tolist(), np.full((3, 5), float('nan')).tolist()])
def test_nonbinary_candidate_fails_the_whole_reply(whole, value):
    candidates, info = llm.parse_response(response([dict(u=whole.result.u.tolist()), dict(u=value)]), whole.d)
    assert not candidates and info['status'] == 'FAILED' and info['accepted'] == 0


def test_too_many_candidates_are_rejected(whole):
    candidates, info = llm.parse_response(response([dict(u=whole.result.u.tolist())]*5), whole.d)
    assert not candidates and info['status'] == 'FAILED'
