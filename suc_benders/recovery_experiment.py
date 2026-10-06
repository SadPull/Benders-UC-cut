"""Independent, serial recovery of the four incomplete Astra repetitions."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import psutil

from .minimal_experiment import configuration
from .subagent import claim, initialize, publish, receipt, sha


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "experiments/rts24-minimal-20261003"
ORDER = (("C", 17), ("D", 7), ("C", 27), ("D", 17))


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def completed_agent(events, exitcode, output_exists):
    identity = next((e["thread_id"] for e in events if e.get("type") == "thread.started"), None)
    completed = any(e.get("type") == "turn.completed" for e in events)
    failed = any(e.get("type") == "turn.failed" for e in events)
    return identity, bool(identity and completed and not failed and exitcode == 0 and output_exists)


def agent_events(path):
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The last event can still be in flight.
    return events


def call_open(run, call, deadline):
    return (time.perf_counter() < deadline and not (run / "cancel.json").exists()
            and not (call / "consumed.json").exists() and not (call / "receipt.json").exists())


def stop_process_tree(process):
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        process.wait(timeout=5)
        return
    children = parent.children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def cli_command(codex, call):
    return [str(codex), "exec", "--ephemeral", "--json",
            "--skip-git-repo-check", "--sandbox", "workspace-write", "--cd", str(call),
            "--model", "gpt-6-astra", "-c", 'model_reasoning_effort="max"',
            "-c", 'model_provider="openai"',
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", "mcp_servers.node_repl.enabled=false", "-c", "memories.use_memories=false",
            "-c", "memories.generate_memories=false", "-c", "notify=[]",
            "-c", "project_doc_max_bytes=0", "--disable", "multi_agent",
            "--disable", "plugins", "--disable", "apps", "--disable", "browser_use",
            "--disable", "computer_use", "--enable", "skip_host_skill_discovery",
            "--output-last-message", str(call / "agent-final.txt"), "-"]


def worker_environment():
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("CODEX_THREAD_ID", None)
    env.pop("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", None)
    # The desktop tool runtime can inject a relay key. exec gives it precedence over saved OAuth.
    env.pop("CODEX_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    # On Windows the app and urllib read the configured system proxy; reqwest needs env vars.
    for scheme in ("http", "https"):
        proxy = urllib.request.getproxies().get(scheme)
        if proxy:
            env.setdefault(f"{scheme.upper()}_PROXY", proxy)
    return env


def model_availability():
    """Read the saved-account model allowance without starting a model session."""
    auth = read(Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json")
    tokens = auth["tokens"]
    request = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
        "Authorization": "Bearer " + tokens["access_token"],
        "ChatGPT-Account-Id": tokens["account_id"], "User-Agent": "codex-cli/0.160.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        usage = json.loads(response.read())
    allowance = usage.get("model_usage", {}).get("gpt-6-astra", {})
    return dict(checked_at=time.time(), available=allowance.get("available"),
        available_at=allowance.get("available_at"),
        account_sha256=hashlib.sha256(tokens["account_id"].encode()).hexdigest())


def check_session_budget(directory):
    manifest = read(directory / "manifest.json")
    prior = manifest.get("failed_startup_calls", 0) + manifest.get("interrupted_recovery_calls", 0)
    launched = len(list(directory.glob("*/calls/*/agent-launch.json")))
    if (launched >= manifest["max_llm_calls_batch"] or
            prior + launched >= manifest.get("authorized_recovery_call_limit", 12)):
        raise RuntimeError("authorized model session quota exhausted; no new session launched")


def serve_call(run, call, codex, deadline, env):
    check_session_budget(run.parent)
    availability = model_availability()
    publish(call / "model-availability.json", availability)
    if availability["available"] is not True:
        raise RuntimeError("Astra allowance unavailable; no new session launched")
    if not call_open(run, call, deadline):
        publish(call / "dispatch-skipped.json", dict(reason="deadline_or_closed_after_preflight", recorded_at=time.time()))
        return dict(receipt_published=False, actual_turn_completed=False)
    iteration = read(call / "request.json")["iteration"]
    instruction = claim(run, iteration)
    (call / "agent-instruction.txt").write_text(instruction, encoding="utf-8")
    command = cli_command(codex, call)
    publish(call / "agent-launch.json", dict(command=command, launched_at=time.time(),
        model="gpt-6-astra", reasoning_effort="max", fresh_session=True))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (call / "agent-events.jsonl").open("w", encoding="utf-8") as stdout, \
         (call / "agent-stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                   cwd=call, env=env, creationflags=flags)
        process.stdin.write(instruction.encode("utf-8"))
        process.stdin.close()
        publish(call / "agent-process.json", dict(pid=process.pid, launched_at=time.time()))
        try:
            while process.poll() is None and call_open(run, call, deadline):
                if (run.parent / "cancel.json").exists():
                    publish(run / "cancel.json", dict(reason="batch_cancelled"))
                    break
                time.sleep(.2)
        finally:
            stop_process_tree(process)
    events = agent_events(call / "agent-events.jsonl")
    identity, success = completed_agent(events, process.returncode, (call / "candidates.json").exists())
    published = False
    if call_open(run, call, deadline):
        if success:
            # Do not edit, repair, or replace the model's JSON before the frozen parser sees it.
            receipt(run, iteration, identity, "completed")
        else:
            receipt(run, iteration, identity or f"codex-cli-process-{process.pid}", "failed")
        published = True
    usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
    completion = dict(agent_id=identity, exitcode=process.returncode,
        actual_turn_completed=success, receipt_published=published, usage=usage,
        late_or_closed=not published, finished_at=time.time(),
        failure=next((e.get("error") for e in reversed(events) if e.get("type") == "turn.failed"), None))
    publish(call / "agent-completion.json", completion)
    return completion


def prepare(directory, codex):
    old = read(ORIGINAL / "manifest.json")
    if any(sha(ORIGINAL / "snapshot" / p) != h for p, h in old["source_sha256"].items()):
        raise ValueError("original frozen snapshot mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    for name in old["source_sha256"]:
        target = directory / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL / "snapshot" / name, target)
    legacy = read(ORIGINAL / "final-summary.json")
    manifest = dict(original_directory=str(ORIGINAL), order=ORDER, budget_seconds=1800,
        source_sha256=old["source_sha256"], packages=old["packages"], executable=sys.executable,
        max_llm_calls_batch=12, max_llm_calls_per_run=3, max_candidates_per_call=4,
        model="gpt-6-astra", reasoning_effort="max", codex_executable=str(codex),
        original_completed_attempts=legacy["attempted_runs"], original_llm_started=legacy["llm_started"],
        reference_sha256=sha(ORIGINAL / "reference/result.json"),
        original_analysis_sha256=sha(ORIGINAL / "final-summary.json"),
        original_file_sha256={p.relative_to(ORIGINAL).as_posix(): sha(p)
                              for p in ORIGINAL.rglob("*") if p.is_file() and "__pycache__" not in p.parts},
        created_at=time.time(), authorized_by_user=True,
        authorization="2026-10-04: rerun interrupted experiments; independent background Astra/max subagents selected",
        protocol_change="Fresh ephemeral codex exec sessions replace chat-bound dispatch; original claim, inputs and receipts retained",
        selection="Replace these four prespecified group/seed runs irrespective of costs; retain all original attempts")
    publish(directory / "manifest.json", manifest)
    return manifest


def supervise(directory, codex, resume=False):
    from .recovery_report import report

    manifest = read(directory / "manifest.json")
    env = worker_environment()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for position, (group, seed) in enumerate(manifest["order"]):
        for name, digest in manifest["source_sha256"].items():
            if sha(directory / "snapshot" / name) != digest:
                raise ValueError(f"recovery snapshot changed: {name}")
        run = directory / f"{group}-{seed}"
        if run.exists():
            if (resume and read(run / "state.json").get("status") in {"TIME_LIMIT", "LLM_CALL_LIMIT", "EVALUATION_LIMIT", "DUAL_CONVERGED", "INNER_NOT_CONVERGED", "MAX_ITERATIONS"}
                    and (run / "result.json").exists()):
                report(directory)
                continue
            raise ValueError(f"run already exists: {run}; preserve it and inspect before another attempt")
        availability = model_availability()
        publish(directory / f"availability-before-{group}-{seed}.json", availability)
        if availability["available"] is not True:
            raise RuntimeError("Astra allowance unavailable; search clock not started")
        initialize(run, dict(group=group, seed=seed, configuration=asdict(configuration(group, seed, run))))
        command = [sys.executable, "-u", "-m", "suc_benders.minimal_experiment", "--worker", group,
                   "--seed", str(seed), "--output-dir", str(run), "--rule-max-calls", "3"]
        started = time.perf_counter()
        with (run / "console.log").open("w", encoding="utf-8") as log:
            worker = subprocess.Popen(command, cwd=directory / "snapshot", env=env,
                                      stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            publish(directory / "state.json", dict(status="running", current=run.name,
                position=position+1, total=len(manifest["order"]), controller_pid=os.getpid(), worker_pid=worker.pid,
                started_at=time.time()))
            try:
                served = set()
                while worker.poll() is None:
                    ready = read(run / "ready.json")
                    deadline = ready.get("deadline_clock", started+60)
                    if not ready and time.perf_counter() >= started+60:
                        raise TimeoutError("worker did not start within 60 seconds")
                    if time.perf_counter() >= deadline+300:
                        raise TimeoutError("worker exceeded deadline plus serialization allowance")
                    if (directory / "cancel.json").exists():
                        publish(run / "cancel.json", {})
                    for request in sorted((run / "calls").glob("*/request.json")):
                        call = request.parent
                        if call not in served and call_open(run, call, deadline):
                            if len(served) >= 3:
                                raise ValueError("call quota exceeded")
                            completion = serve_call(run, call, codex, deadline, env)
                            served.add(call)
                            if completion["receipt_published"] and not completion["actual_turn_completed"]:
                                raise RuntimeError("independent model session failed; preserve attempt and stop remaining runs")
                    time.sleep(.2)
            except BaseException as exc:
                publish(run / "cancel.json", dict(reason="controller_failure", recorded_at=time.time()))
                stop_process_tree(worker)
                publish(run / "interruption.json", dict(reason=str(exc), recorded_at=time.time(),
                    worker_pid=worker.pid, worker_exitcode=worker.returncode,
                    state_before_interruption=read(run / "state.json")))
                raise
        state = read(run / "state.json")
        state.update(exitcode=worker.returncode, process_wall_seconds=time.perf_counter()-started)
        publish(run / "state.json", state)
        if worker.returncode != 0 or state.get("status") not in {"TIME_LIMIT", "LLM_CALL_LIMIT", "EVALUATION_LIMIT", "DUAL_CONVERGED", "INNER_NOT_CONVERGED", "MAX_ITERATIONS"}:
            raise RuntimeError(f"worker failed: {run.name}; remaining runs not started")
        print(f"{position+1}/{len(manifest['order'])} {run.name}: {state['status']}", flush=True)
        report(directory)
        if (directory / "cancel.json").exists():
            publish(directory / "state.json", dict(status="cancelled", controller_pid=os.getpid()))
            return
    publish(directory / "state.json", dict(status="completed", runs=len(manifest["order"]), completed_at=time.time()))
    report(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codex", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip normally ended runs; never redispatch an existing call")
    args = parser.parse_args()
    directory, codex = Path(args.output_dir).resolve(), Path(args.codex).resolve()
    if not args.prepared:
        prepare(directory, codex)
    if args.prepare_only:
        return
    if read(directory / "manifest.json").get("execution_authorization_status") == "pending_call_limit_decision":
        raise PermissionError("call limit decision pending; no search or model session started")
    try:
        supervise(directory, codex, args.resume)
    except BaseException as exc:
        publish(directory / "state.json", dict(status="failed", error_type=type(exc).__name__,
            error=str(exc), controller_pid=os.getpid(), finished_at=time.time()))
        raise


if __name__ == "__main__":
    main()
