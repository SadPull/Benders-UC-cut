"""Fixed 5 x 50 study driven by fresh Codex subagents in the supervising chat.

The local transport uses a synthetic Responses envelope solely to reuse the
existing parser and repair. It makes no HTTP requests and does not launch agents.
The chat dispatches one agent per request and writes a receipt after it finishes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from . import efficiency_experiment as study
from .attribution import write_json
from .data import load_suc_data
from .llm import ask_llm_for_candidates
from .solver import result_to_dict, solve_benders


MODEL = "gpt-6-astra"
ROUNDS = 50
OLD = study.ROOT / "results/llm-sa-efficiency-50rounds-20261001-180425"
ORIGINAL_CONFIGURATION = study.configuration


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(seed, strategy):
    config = ORIGINAL_CONFIGURATION(seed, strategy, rounds=ROUNDS)
    if strategy == "sa_llm":
        config = replace(config, llm=replace(config.llm, api_format="responses",
            base_url="https://offline.invalid", model=MODEL, max_tokens=None))
    return config


class SubagentTransport:
    def __init__(self, path, seed):
        self.path, self.seed = path, seed
        self.receipt = None

    def open(self, request, timeout=None):
        payload = json.loads(request.data)
        assert payload["model"] == MODEL and payload["reasoning"] == {"effort": "max"}
        assert timeout is None and "max_output_tokens" not in payload
        system, user = (item["content"] for item in payload["input"])
        iteration = json.loads(user)["iteration"]
        call = self.path / "calls" / f"{iteration:03d}"
        call.mkdir(parents=True, exist_ok=False)
        (call / "system.txt").write_bytes(system.encode("utf-8"))
        (call / "user.txt").write_bytes(user.encode("utf-8"))
        hashes = {name: sha(call / name) for name in ("system.txt", "user.txt")}
        write_json(call / "request.json", {"seed": self.seed, "iteration": iteration,
            "model": MODEL, "reasoning_effort": "max", "fork_turns": "none",
            "max_candidates": 4, "input_sha256": hashes, "created_at": now(),
            "transport": "codex_subagent", "temperature_override": None,
            "max_output_tokens_override": None})
        print(f"WAIT_SUBAGENT s{self.seed} round={iteration} {call}", flush=True)
        while not (call / "receipt.json").exists():
            time.sleep(.5)
        receipt = json.loads((call / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["model"] == MODEL and receipt["reasoning_effort"] == "max"
        assert receipt["input_sha256"] == hashes
        assert receipt["subagent_calls"] == 1 and receipt["agent_id"]
        assert all(sha(call / name) == value for name, value in hashes.items())
        content = (call / "candidates.json").read_bytes()
        assert hashlib.sha256(content).hexdigest() == receipt["output_sha256"]
        self.receipt = receipt
        # This envelope is generated locally, never returned by an HTTP endpoint.
        envelope = {"status": "completed", "model": MODEL, "output": [
            {"type": "message", "content": [{"type": "output_text", "text": content.decode("utf-8")}]}]}
        return io.BytesIO(json.dumps(envelope).encode())


def adviser(path, seed):
    def ask(*args, **kwargs):
        transport = SubagentTransport(path, seed)
        # The sentinel replaces any inherited credential. No credential file is read.
        with patch.dict(os.environ, {"LLM_API_KEY": "OFFLINE_SUBAGENT_TRANSPORT"}), \
             patch("suc_benders.llm.build_opener", return_value=transport):
            advice = ask_llm_for_candidates(*args, **kwargs)
        record = advice.diagnostics
        record.update(transport="codex_subagent", synthetic_envelope=True,
                      http_requests=0, subagent_receipt=transport.receipt,
                      effective_temperature=None, max_output_tokens_override=None)
        call = path / "calls" / f"{args[2]:03d}"
        write_json(call / "diagnostics.json", record)
        return advice
    return ask


def prepare(directory):
    # Existing fingerprinting covers the complete solver, input, packages and config.
    with patch.object(study, "configuration", side_effect=lambda seed, strategy, rounds=None: configuration(seed, strategy)):
        study.prepare(directory, False, rounds=ROUNDS)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data"] == old["data"] and manifest["scenarios"] == old["scenarios"]
    assert manifest["configurations"]["sa"] == old["configurations"]["sa"]
    assert manifest["packages"] == old["packages"]
    # Only the API client and CLI had changed before this experiment; SA code is identical.
    for name, expected in old["sources"].items():
        if name not in ("suc_benders/llm.py", "suc_benders/run.py"):
            assert manifest["sources"][name] == expected, name
    origin = OLD / "runs/sa__s7"
    assert json.loads((origin / "state.json").read_text())["status"] == "completed"
    events = study.read_events(origin / "events.jsonl")
    assert sum(e["kind"] == "iteration" for e in events) == ROUNDS
    shutil.copytree(origin, directory / "runs/sa__s7")
    write_json(directory / "protocol.json", {"model": MODEL, "reasoning_effort": "max",
        "rounds": ROUNDS, "seeds": study.SEEDS, "planned_subagent_calls": 250,
        "serial": True, "fresh_agent_per_call": True, "automation": False,
        "max_output_tokens_override": None, "temperature_override": None,
        "quantum_hardware_calls": 0, "synthetic_envelope_for_parser_only": True,
        "reference_hidden_from_advisers": True, "reused_sa_seed7": str(origin),
        "reused_events_sha256": sha(origin / "events.jsonl"),
        "old_events_sha256": {str(p.relative_to(OLD)): sha(p) for p in (OLD / "runs").glob("*/events.jsonl")},
        "prepared_at": now()})


def run_one(directory, seed, strategy):
    path = directory / "runs" / f"{strategy}__s{seed}"
    if path.exists():
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if state["status"] != "completed":
            raise RuntimeError(f"Incomplete run retained; no automatic replay: {path.name}")
        print(f"SKIP completed {path.name}", flush=True)
        return
    path.mkdir(parents=True)
    state = {"status": "running", "seed": seed, "strategy": strategy,
             "budget": None, "rounds": ROUNDS, "started_at": now(), "pid": os.getpid()}
    write_json(path / "state.json", state)
    print(f"START {path.name}", flush=True)
    data, scenarios = load_suc_data(study.ROOT / "data/suc_10x24.json")
    with (path / "events.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
        def event(row):
            stream.write(json.dumps(result_to_dict(row), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if row["kind"] == "iteration":
                print(f"ROUND {path.name} {row['iteration']}/{ROUNDS} cost={row['incumbent']:.6f}", flush=True)
                write_json(path / "progress.json", {**row, "updated_at": now()})
        config = replace(configuration(seed, strategy), event_sink=event)
        try:
            with patch("suc_benders.paper_solver.ask_llm_for_candidates", side_effect=adviser(path, seed)):
                result = solve_benders(data, scenarios, config)
            write_json(path / "result.json", result)
            events = study.read_events(path / "events.jsonl")
            assert sum(e["kind"] == "iteration" for e in events) == ROUNDS
            assert sum(e["kind"] == "sa_start" for e in events) == ROUNDS
            assert sum(e["kind"] == "llm_start" for e in events) == (ROUNDS if strategy == "sa_llm" else 0)
            state.update(status="completed", ended_at=now(), exitcode=0)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error_type=type(exc).__name__, ended_at=now())
            raise
        finally:
            write_json(path / "state.json", state)
    print(f"DONE {path.name}", flush=True)


def report(directory):
    # Call only between runs: the original report mutates active states.
    study.report(directory)
    path = directory / "report.md"
    content = path.read_text(encoding="utf-8")
    content = content.replace("LLM请求max与65536 tokens；接口接受请求，但未能独立确认后端实际推理强度。",
        "LLM 全部使用全新 gpt-6-astra 子代理、max 推理；未设置 token 上限或温度覆盖，实际 token 用量未知。"
        "本地 Responses 封装仅复用解析和修复，无外部 HTTP 请求。时间包含主聊天调度、子代理读写与等待。")
    content += "\n种子 7 的 SA 对照来自旧实验，其余运行在本次执行。旧 DeepSeek 轨迹不并入 Astra 结果。相对参考差距不是认证 gap。量子真机调用为零。\n"
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if args.prepare_only:
        prepare(directory)
        return
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha(study.ROOT / name) == value for name, value in manifest["sources"].items())
    with (directory / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": now()}, stream)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import gurobipy as gp
    gp.setParam("Threads", 1)
    for index, seed in enumerate(study.SEEDS):
        for strategy in (("sa", "sa_llm") if index % 2 == 0 else ("sa_llm", "sa")):
            run_one(directory, seed, strategy)
            report(directory)
    write_json(directory / "completion.json", {"completed_at": now(), "runs": 10, "subagent_calls": 250})


if __name__ == "__main__":
    main()
