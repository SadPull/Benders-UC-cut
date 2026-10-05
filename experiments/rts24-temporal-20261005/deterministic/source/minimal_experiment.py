"""Frozen, serial five-arm RTS-24 necessity experiment (15 searches + reference)."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

from .data import load_suc_data
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders
from .subagent import initialize, publish, sha


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"A": "SA fixed seed", "B": "SA changing seed", "C": "LLM only",
          "D": "SA + LLM", "E": "SA + rule heuristic"}
SEEDS = (7, 17, 27)
BUDGET = 1800.


def configuration(group, seed, directory, rule_max_calls=3):
    llm = LLMConfig(enabled=group in {"C", "D"}, api_format="codex_subagent",
        model="gpt-6-astra", reasoning_effort="max", max_tokens=None,
        timeout_seconds=None, max_retries=1, max_candidates=4,
        collect_diagnostics=True, enhanced_feedback=True, exchange_dir=str(directory),
        max_calls=rule_max_calls if group == "E" else 3,
        schedule="fixed" if group == "C" else "stagnation_burst")
    return BendersConfig(max_iterations=20, time_limit_seconds=BUDGET,
        advisor_only=group == "C",
        quantum_candidates=group != "C", sa_seed_per_call=group == "B",
        rule_advisor=group == "E", quantum=QuantumConfig(kaiwu_seed=seed, candidate_pool_size=16),
        llm=llm)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(directory, data, scenarios):
    from .full_model import solve_full, FullConfig
    from .subproblem import solve_scenario
    from .validation import validate_schedule, schedule_object
    started = time.perf_counter()
    result = solve_full(data, scenarios, FullConfig(time_limit=600, mip_gap=1e-6, output_flag=1))
    raw = result_to_dict({k: v for k, v in vars(result).items() if k != "model"})
    raw["solver_runtime_seconds"] = result.model.Runtime
    publish(directory / "raw.json", raw)
    result.model.dispose()
    phase = time.perf_counter()
    checked = [solve_scenario(data, s, result.u, result.v, result.w,
        deadline=time.perf_counter()+60, threads=1) for s in scenarios]
    if not all(r.feasible for r in checked):
        raise ValueError("reference fixed-dispatch check failed")
    verified = dict(raw)
    verified["raw_objective"] = raw["objective"]
    verified["expected_fuel_cost"] = sum(s.probability*r.objective for s, r in zip(scenarios, checked))
    verified["objective"] = raw["fixed_cost"]+verified["expected_fuel_cost"]
    verified["scenarios"] = result_to_dict([dict(name=s.name, probability=s.probability,
        p=r.p, r=r.r, fuel_cost=r.objective) for s, r in zip(scenarios, checked)])
    audit = validate_schedule(data, scenarios, schedule_object(verified))
    difference = abs(raw["objective"]-verified["objective"])
    if difference > 1e-3+1e-7*abs(raw["objective"]):
        raise ValueError("reference objective inconsistent with fixed-dispatch check")
    audit.update(fixed_dispatch_seconds=time.perf_counter()-phase, objective_difference=difference)
    publish(directory / "result.json", verified)
    publish(directory / "audit.json", audit)
    publish(directory / "state.json", dict(status="completed", elapsed_seconds=time.perf_counter()-started))


def worker(directory, group, seed, rule_max_calls):
    import gurobipy as gp
    gp.setParam("Threads", 1)
    gp.setParam("Seed", seed)
    data, scenarios = load_suc_data(ROOT / "data/rts24/units.json", ROOT / "data/rts24/stochastic.json")
    started = time.perf_counter()
    publish(directory / "state.json", dict(status="running", pid=os.getpid()))
    try:
        if group == "reference":
            reference(directory, data, scenarios)
            return
        config = configuration(group, seed, directory, rule_max_calls)
        with (directory / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
            def event(row):
                serialized = result_to_dict(row)
                stream.write(json.dumps(serialized, ensure_ascii=False, allow_nan=False)+"\n")
                if row["kind"] == "started":
                    publish(directory / "ready.json", serialized)
                if row["kind"] == "evaluation" and row["improved"]:
                    publish(directory / "best-checkpoint.json", serialized)
                if row["kind"] in {"iteration", "llm_start", "rule_start", "sa_start"}:
                    publish(directory / "progress.json", serialized)
            config.event_sink = event
            result = solve_benders(data, scenarios, config)
            publish(directory / "result.json", result_to_dict(result))
        publish(directory / "state.json", dict(status=result.status, elapsed_seconds=result.runtime))
    except BaseException as exc:
        publish(directory / "state.json", dict(status="failed", error_type=type(exc).__name__,
            elapsed_seconds=time.perf_counter()-started))
        raise


def freeze(directory, rule_max_calls):
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "snapshot"
    paths = [*ROOT.glob("suc_benders/*.py"), *ROOT.glob("tests/*.py"),
             *ROOT.glob("data/rts24/*.json"), ROOT / "requirements.txt"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = sha(target)
    order = [(group, seed) for group in GROUPS for seed in SEEDS]
    random.Random(20261003).shuffle(order)
    manifest = dict(created_at=time.time(), groups=GROUPS, seeds=SEEDS, order=order,
        shuffle_seed=20261003, budget_seconds=BUDGET, rule_max_calls=rule_max_calls,
        max_llm_calls_per_run=3, max_llm_calls_batch=18, max_candidates_per_call=4,
        reference=dict(time_limit=600, mip_gap=1e-6, threads=1, seed=7),
        source_sha256=hashes, python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "gurobipy", "kaiwu")},
        protocol=dict(serial=True, initialization_in_budget=True, reference_not_search_input=True,
            late_candidates_excluded=True, target_stopping=False, iteration_limit=None,
            llm_fresh_sessions=True, llm_usage="unknown unless provided by model tool",
            rule_scoring="same rounded top-60 gradient; missing fuel changes assumed zero; transition costs exact",
            rule_candidates="all ON blocks and proper prefixes/suffixes ranked; repair/skip duplicates until four unique schedules"))
    publish(directory / "manifest.json", manifest)
    return manifest


def supervise(directory, manifest):
    from .minimal_report import report
    snapshot = directory / "snapshot"
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    runs = [("reference", 7), *manifest["order"]]
    for position, (group, seed) in enumerate(runs):
        name = "reference" if group == "reference" else f"{group}-{seed}"
        run = directory / name
        if run.exists():
            state = read(run / "state.json") if (run / "state.json").exists() else {}
            if state.get("status") == "running":
                raise ValueError(f"unfinished run {name}; inspect its process before resuming")
            continue
        for path, digest in manifest["source_sha256"].items():
            if sha(snapshot / path) != digest:
                raise ValueError(f"frozen source/input changed: {path}")
        config = configuration(group, seed, run, manifest["rule_max_calls"]) if group != "reference" else None
        initialize(run, dict(group=group, seed=seed, configuration=asdict(config) if config else manifest["reference"]))
        started = time.perf_counter()
        with (run / "console.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-m", "suc_benders.minimal_experiment", "--worker", group,
                       "--seed", str(seed), "--output-dir", str(run),
                       "--rule-max-calls", str(manifest["rule_max_calls"] or 0)]
            process = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
            publish(directory / "state.json", dict(status="running", current=name,
                position=position, total=len(runs), controller_pid=os.getpid(), worker_pid=process.pid,
                started_at=time.time()))
            try:
                process.wait(timeout=900 if group == "reference" else BUDGET+300)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=20)
                publish(run / "state.json", dict(status="worker_timeout"))
        state = read(run / "state.json") if (run / "state.json").exists() else dict(status="failed")
        if state.get("status") == "running":
            state["status"] = "failed"
        state.update(exitcode=process.returncode, process_wall_seconds=time.perf_counter()-started)
        publish(run / "state.json", state)
        print(f"{position+1}/{len(runs)} {name}: {state['status']}", flush=True)
        report(directory)
    publish(directory / "state.json", dict(status="completed", runs=len(runs), completed_at=time.time()))
    report(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", choices=[*GROUPS, "reference"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rule-max-calls", type=int, default=3, help="0 means unlimited rule bursts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if args.worker:
        worker(directory, args.worker, args.seed, args.rule_max_calls or None)
    elif args.report:
        from .minimal_report import report
        report(directory)
    else:
        manifest = read(directory / "manifest.json") if args.resume else freeze(directory, args.rule_max_calls or None)
        supervise(directory, manifest)


if __name__ == "__main__":
    main()
