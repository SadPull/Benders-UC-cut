"""File exchange for Codex-controlled Astra calls. No model or HTTP client."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid


MODEL = "gpt-6-astra"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def cancelled(config):
    return bool(config.exchange_dir and (Path(config.exchange_dir) / "cancel.json").exists())


def exchange(config, iteration, system, user):
    if not config.exchange_dir:
        raise ValueError("codex_subagent requires exchange_dir")
    if config.model not in {None, MODEL} or config.reasoning_effort != "max" or not 1 <= config.max_candidates <= 4:
        raise ValueError("codex_subagent requires gpt-6-astra/max and 1-4 candidates")
    root = Path(config.exchange_dir)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    call = root / "calls" / f"{iteration:03d}"
    call.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + config.timeout_seconds if config.timeout_seconds is not None else float("inf")
    call_id = f"{run['run_id']}:{iteration}"
    record = dict(call_id=call_id, transport="codex_subagent", http_requests=0,
                  requested_model=MODEL, reasoning_effort="max", usage=None,
                  effective_temperature=None, max_output_tokens_override=None)
    for name, content in (("system.txt", system), ("user.txt", user)):
        (call / name).write_bytes(content.encode("utf-8"))
    hashes = {name: sha(call / name) for name in ("system.txt", "user.txt")}
    publish(call / "request.json", dict(run_id=run["run_id"], call_id=call_id,
            iteration=iteration, model=MODEL, reasoning_effort="max", fork_turns="none",
            max_candidates=config.max_candidates, input_sha256=hashes,
            created_at=time.time(), status="pending"))
    print(f"WAIT_SUBAGENT {call.resolve()}", flush=True)
    while True:
        status = "cancelled" if cancelled(config) else "deadline" if time.perf_counter() >= deadline else None
        if status:
            break
        receipt_path = call / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = dict(run_id=run["run_id"], call_id=call_id, model=MODEL,
                            reasoning_effort="max", input_sha256=hashes)
            if any(receipt.get(k) != v for k, v in expected.items()) or any(sha(call / k) != v for k, v in hashes.items()):
                raise ValueError("subagent receipt/input integrity mismatch")
            if receipt.get("status") not in {"completed", "failed"} or not receipt.get("agent_id"):
                raise ValueError("invalid subagent receipt")
            status = receipt["status"]
            record["receipt"] = receipt
            if status == "completed":
                output = call / "candidates.json"
                if sha(output) != receipt.get("output_sha256"):
                    raise ValueError("subagent output integrity mismatch")
                content = output.read_text(encoding="utf-8")
                break
            break
        time.sleep(.2)
    record.update(status=status, agent_wait_seconds=time.perf_counter()-started)
    publish(call / "consumed.json", record)
    return (content if status == "completed" else None), record


def initialize(root, manifest):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    publish(root / "run.json", dict(manifest, run_id=uuid.uuid4().hex, created_at=time.time()))


def claim(root, iteration):
    """Reserve one dispatch and return its program-generated agent instruction."""
    root = Path(root).resolve()
    call = root / "calls" / f"{iteration:03d}"
    request = json.loads((call / "request.json").read_text(encoding="utf-8"))
    if (root / "cancel.json").exists() or (call / "consumed.json").exists() or (call / "receipt.json").exists():
        raise ValueError("call is closed")
    with (call / "dispatch.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(call_id=request["call_id"], claimed_at=time.time()), stream)
    return (f"Read ONLY {call / 'system.txt'} and {call / 'user.txt'}. "
            f"Follow those business instructions. Write the final candidate JSON to {call / 'candidates.json'}. "
            "Use file-reading and file-writing tools only. Do not read other files or reference results, "
            "invoke solvers or external APIs, execute optimization code, or create other agents. "
            "Do not modify feedback or receipt files. Report completion after writing the JSON.")


def receipt(root, iteration, agent_id, status):
    """Called by Codex after waiting for the actual agent to finish."""
    call = Path(root) / "calls" / f"{iteration:03d}"
    if (call / "consumed.json").exists() or (call / "receipt.json").exists():
        raise ValueError("call is closed")
    request = json.loads((call / "request.json").read_text(encoding="utf-8"))
    value = {k: request[k] for k in ("run_id", "call_id", "model", "reasoning_effort", "input_sha256")}
    value.update(agent_id=agent_id, status=status, completed_at=time.time())
    if status == "completed":
        value["output_sha256"] = sha(call / "candidates.json")
    publish(call / "receipt.json", value)
