"""File exchange between the experiment worker and Codex-managed fresh agents."""
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep, time
from uuid import uuid4

from suc_benders.llm import parse_response, request_body


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def exchange(directory, instructions, user, seconds, kind):
    directory.mkdir(parents=True, exist_ok=False)
    request_id = uuid4().hex
    started = perf_counter()
    deadline = started + seconds
    (directory / "system.txt").write_text(instructions, encoding="utf-8")
    (directory / "user.json").write_text(user, encoding="utf-8")
    save(directory / "request.json", dict(request_id=request_id, kind=kind,
         model="gpt-6-astra", reasoning_effort="max", created_at=time(), deadline_at=time()+seconds,
         timeout_seconds=seconds, response_file="response.json", receipt_file="receipt.json"))
    print(f"CODEX_REQUEST {directory.resolve()}", flush=True)
    try:
        while perf_counter() < deadline:
            if (directory / "receipt.json").exists():
                receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
                response = (directory / "response.json").read_bytes()
                if (receipt["request_id"] != request_id or not receipt.get("agent_id")
                        or receipt.get("status") != "completed"
                        or receipt.get("model") != "gpt-6-astra"
                        or receipt.get("reasoning_effort") != "max"
                        or receipt["response_sha256"] != hashlib.sha256(response).hexdigest()):
                    raise ValueError("Invalid completion receipt")
                payload = json.loads(response)
                if perf_counter() >= deadline:
                    break
                save(directory / "consumed.json", dict(status="ACCEPTED", request_id=request_id,
                     agent_id=receipt["agent_id"], elapsed_seconds=perf_counter()-started))
                return payload, receipt
            sleep(min(.2, max(0, deadline-perf_counter())))
        raise TimeoutError("Subagent response deadline")
    except Exception as error:
        save(directory / "consumed.json", dict(status="FAILED", request_id=request_id,
             error_type=type(error).__name__, elapsed_seconds=perf_counter()-started))
        raise


def preflight(o, directory):
    started = perf_counter()
    result = dict(status="FAILED", transport="codex_subagent", model=o.llm_model,
                  reasoning_effort="max", counts_toward_search_calls=False)
    try:
        payload, receipt = exchange(directory / "connectivity", "Reply with OK only.",
                                    json.dumps({"task": "Connectivity check; return {\"text\":\"OK\"}."}),
                                    o.llm_timeout, "connectivity")
        if not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise ValueError("Empty subagent response")
        result.update(status="PASSED", agent_id=receipt["agent_id"], response_characters=len(payload["text"]))
    except Exception as error:
        result["error_type"] = type(error).__name__
    result["elapsed_seconds"] = perf_counter() - started
    return result


def request(d, scenarios, best, o, seconds, context, base_candidates, iteration, *, directory):
    started = perf_counter()
    call_dir = Path(directory) / f"{iteration:03d}"
    info = dict(status="FAILED", message="", accepted=0, format_valid=0, proposals=[],
                transport="codex_subagent")
    candidates = []
    try:
        body = request_body(d, scenarios, best, o, context, base_candidates, iteration)
        payload, receipt = exchange(call_dir, body["instructions"], body["input"],
                                    min(seconds, o.llm_timeout), "candidates")
        wrapped = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(payload)}]}]}
        candidates, parsed = parse_response(wrapped, d, iteration)
        info.update(parsed, agent_id=receipt["agent_id"])
    except Exception as error:
        info["message"] = type(error).__name__
    info["seconds"] = perf_counter() - started
    if info["seconds"] >= seconds:
        candidates = []
        info.update(status="TIME_LIMIT", accepted=0)
    return candidates, info
