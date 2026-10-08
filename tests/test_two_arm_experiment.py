import hashlib
import json
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

import run_two_arm_experiment as experiment
import subagent_advisor
from suc_benders.config import Options


def configured():
    return Options(llm_enabled=True, llm_model="test-model", llm_url="https://example.invalid/responses",
                   llm_api_key=uuid4().hex, kaiwu_user_id=uuid4().hex, kaiwu_sdk_code=uuid4().hex)


@pytest.mark.parametrize("payload,expected", [({"text": "OK"}, "PASSED"), ({"text": ""}, "FAILED")])
def test_preflight_uses_subagent(monkeypatch, tmp_path, payload, expected):
    o = configured()
    calls = []

    def transport(directory, instructions, user, seconds, kind):
        calls.append((seconds, kind))
        return payload, {"agent_id": "preflight-agent"}

    monkeypatch.setattr(subagent_advisor, "exchange", transport)
    result = subagent_advisor.preflight(o, tmp_path)
    assert result["status"] == expected
    assert len(calls) == 1
    assert calls[0] == (o.llm_timeout, "connectivity")
    assert result["transport"] == "codex_subagent"


def test_failed_preflight_stops_every_solver_and_redacts_error(monkeypatch, tmp_path):
    o = configured()
    calls = []

    def transport(directory, instructions, user, seconds, kind):
        calls.append(kind)
        raise TimeoutError(o.llm_api_key)

    def forbidden(*args, **kwargs):
        pytest.fail("No solver or snapshot may run after failed LLM preflight")

    monkeypatch.setattr(experiment, "load_options", lambda overrides: o)
    monkeypatch.setattr(subagent_advisor, "exchange", transport)
    for name in ("check_kaiwu", "reference", "run_suc", "freeze"):
        monkeypatch.setattr(experiment, name, forbidden)
    output = tmp_path / "failed"
    assert experiment.run_batch(output) == 1
    assert len(calls) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "STOPPED_LLM_PREFLIGHT"
    assert summary["preflight"]["error_type"] == "TimeoutError"
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8")
        assert o.llm_api_key not in text and o.llm_url not in text


def test_snapshot_and_options_exclude_credentials(tmp_path):
    o = configured()
    experiment.freeze(tmp_path, o)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["order"] == ["D-7", "A-7"]
    assert all(key not in manifest["options"] for key in
               ("llm_api_key", "llm_url", "kaiwu_user_id", "kaiwu_sdk_code"))
    assert not list(tmp_path.rglob("config.local.json"))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert all(secret.encode() not in content for secret in
                       (o.llm_api_key, o.kaiwu_user_id, o.kaiwu_sdk_code))


@pytest.mark.parametrize("corrupt", [False, True])
def test_exchange_requires_matching_receipt(monkeypatch, tmp_path, corrupt):
    directory = tmp_path / "call"

    def respond(seconds):
        request = json.loads((directory / "request.json").read_text())
        response = b'{"candidates": []}'
        (directory / "response.json").write_bytes(response)
        subagent_advisor.save(directory / "receipt.json", dict(request_id=request["request_id"],
            agent_id="fresh-agent", status="completed", model="gpt-6-astra", reasoning_effort="max",
            response_sha256="invalid" if corrupt else hashlib.sha256(response).hexdigest()))

    monkeypatch.setattr(subagent_advisor, "sleep", respond)
    if corrupt:
        with pytest.raises(ValueError):
            subagent_advisor.exchange(directory, "system", "{}", 10, "candidates")
    else:
        payload, receipt = subagent_advisor.exchange(directory, "system", "{}", 10, "candidates")
        assert payload == {"candidates": []} and receipt["agent_id"] == "fresh-agent"


def test_exchange_timeout_and_adapter_never_uses_http(monkeypatch, tmp_path):
    import suc_benders.llm as core_llm

    def forbidden(*args, **kwargs):
        pytest.fail("HTTP is forbidden in the Codex experiment")

    monkeypatch.setattr(core_llm, "urlopen", forbidden)
    with pytest.raises(TimeoutError):
        subagent_advisor.exchange(tmp_path / "expired", "system", "{}", .001, "candidates")
    monkeypatch.setattr(subagent_advisor, "request_body", lambda *args: {"instructions": "system", "input": "{}"})
    monkeypatch.setattr(subagent_advisor, "exchange", lambda *args:
                        ({"candidates": [{"u": [[1, 0]]}]}, {"agent_id": "candidate-agent"}))
    rows, info = subagent_advisor.request(NS(n_units=1, horizon=2), [], None, configured(),
                                         10, {}, [], 1, directory=tmp_path)
    assert len(rows) == 1 and info["status"] == "OK"
    assert info["transport"] == "codex_subagent" and info["agent_id"] == "candidate-agent"
