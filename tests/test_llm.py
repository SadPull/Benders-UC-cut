import io
import json

import numpy as np
import pytest

from suc_benders import llm
from suc_benders.benders import dispatch
from suc_benders.config import load_options
from suc_benders.model import model, partition
from suc_benders.validation import pack


def response(rows):
    return {"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"candidates": rows})}]}]}


@pytest.fixture
def context(tiny, options):
    d, s = tiny()
    b = partition(model(d, s), 3)[0][0]
    best = dispatch(b, pack(d, np.ones((2, 3))), 10)
    row = {name: best.point[getattr(b, name)].tolist() for name in ("u", "v", "w")}
    options.llm_url, options.llm_api_key, options.llm_model = "https://example.invalid/v1/responses", "offline-secret", "offline-model"
    return b, best, options, row


def test_request_uses_priced_sparse_context(context):
    b, best, o, _ = context
    body = llm.request_body(b, best, 0, o, {"cuts": [best.cut]})
    assert body["stream"] is False and body["store"] is False
    assert "offline-secret" not in json.dumps(body)
    data = json.loads(body["input"])
    assert data["local_constraints"]["one_based_indices"] is False
    assert data["hours"] == [1, 2, 3]
    triplets = np.asarray(data["local_constraints"]["triplets"])
    from scipy import sparse
    restored = sparse.coo_matrix((triplets[:, 2], (triplets[:, 0].astype(int), triplets[:, 1].astype(int))), shape=b.A.shape).tocsr()
    assert (restored != b.A).nnz == 0
    np.testing.assert_equal(data["priced_continuous_cost"], b.obj[b.continuous])


def test_valid_duplicate_and_malformed_candidates(context):
    b, best, _, row = context
    candidates, info = llm.parse_response(response([{"u": [[.5]]}, row, row]), b)
    assert info["format_valid"] == 2 and info["accepted"] == 1
    np.testing.assert_array_equal(candidates[0], best.point[b.binary])


@pytest.mark.parametrize("bad", [None, {}, {"status": "incomplete"}, {"status": "completed", "output": []},
                                  {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "invalid"}]}]}])
def test_bad_responses(context, bad):
    candidates, info = llm.parse_response(bad, context[0])
    assert not candidates and info["status"] == "FAILED"


@pytest.mark.parametrize("value", [[[1]], [[1, 1, .5], [1, 1, 1]], [[1, 1, float("nan")], [1, 1, 1]],
                                    [["1", "1", "1"], ["1", "1", "1"]]])
def test_bad_matrix(context, value):
    b, _, _, row = context
    row["u"] = value
    candidates, info = llm.parse_response(response([row]), b)
    assert not candidates and info["format_valid"] == 0


def test_transport_and_timeout(context, monkeypatch):
    b, best, o, row = context
    seen = {}

    def post(req, timeout):
        seen.update(body=json.loads(req.data), timeout=timeout)
        return io.BytesIO(json.dumps(response([row])).encode())

    monkeypatch.setattr(llm, "urlopen", post)
    candidates, info = llm.request(b, best, 0, o, 10, {})
    assert candidates and info["status"] == "OK" and 0 < seen["timeout"] <= 10

    def failed(*args, **kwargs):
        raise RuntimeError("offline-secret must not appear in a result")

    monkeypatch.setattr(llm, "urlopen", failed)
    candidates, info = llm.request(b, best, 0, o, 10, {})
    assert not candidates and info["status"] == "FAILED"
    assert "offline-secret" not in str(info)
    monkeypatch.setattr(llm, "urlopen", post)
    times = iter([0., 0., 11., 11.])
    monkeypatch.setattr(llm, "perf_counter", lambda: next(times))
    candidates, info = llm.request(b, best, 0, o, 10, {})
    assert not candidates and info["status"] == "TIME_LIMIT"


def test_local_configuration_and_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"kaiwu": {"user_id": "test", "sdk_code": "test-secret"},
                                "llm": {"enabled": True, "url": "https://example.invalid", "api_key": "test-key", "model": "test"}}))
    o = load_options({"llm_enabled": False}, path)
    assert not o.llm_enabled and o.kaiwu_sdk_code == "test-secret"
    assert "test-secret" not in repr(o) and "test-key" not in repr(o)
    with pytest.raises(ValueError):
        load_options({"llm_api_key": ""}, path)
