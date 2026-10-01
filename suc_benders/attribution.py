"""Fixed, resumable 3-case x 5-repeat x 10-strategy attribution experiment."""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import numpy as np

from .common import commitment_is_feasible, require_gurobi
from .data import Scenario, load_suc_data
from .full_model import FullConfig, solve_full
from .llm import LLMConfig, llm_settings
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 27, 37, 47)
CASES = ("canonical", "light_90", "peak_shift_2h")
STRATEGIES = [
    {"name": f"q{q}_l{l}_d{d}", "q": bool(q), "l": bool(l), "d": bool(d), "repair": True, "random": False}
    for q in (0, 1) for l in (0, 1) for d in (0, 1)
] + [
    {"name": "q_only_raw", "q": True, "l": False, "d": False, "repair": False, "random": False},
    {"name": "random_repair", "q": False, "l": False, "d": False, "repair": True, "random": True},
]
LOGGER = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result_to_dict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def case_scenarios(scenarios, name):
    if name == "canonical":
        return list(scenarios)
    if name == "light_90":
        return [Scenario(s.name, s.probability, s.load * .9) for s in scenarios]
    if name == "peak_shift_2h":
        return [Scenario(s.name, s.probability, np.roll(s.load, 2)) for s in scenarios]
    raise ValueError(f"unknown case: {name}")


def schedule():
    rows = []
    for case in CASES:
        for seed in SEEDS:
            strategies = list(STRATEGIES)
            random.Random(f"attribution-20260930:{case}:{seed}").shuffle(strategies)
            for strategy in strategies:
                rows.append({"run_id": f"{case}__s{seed}__{strategy['name']}",
                             "case": case, "seed": seed, "strategy": strategy})
    return rows


def research_config(strategy, seed, llm_config):
    return BendersConfig(
        max_iterations=10, max_candidate_evaluations=40, collect_diagnostics=True,
        quantum_candidates=strategy["q"], repair_candidates=strategy["repair"],
        decommit_candidates=strategy["d"], decommit_ladder_size=8,
        random_candidates=strategy["random"], random_seed=seed, random_candidate_pool_size=16,
        quantum=QuantumConfig(kaiwu_seed=seed, candidate_pool_size=16),
        llm=LLMConfig(**{**asdict(llm_config), "enabled": strategy["l"]}),
        subproblem_output_flag=0,
    )


def validate_result(data, scenarios, result):
    """Check physical residuals without introducing additional optimization calls."""
    diagnostics = result.diagnostics
    events = diagnostics["evaluations"]
    fresh = [row for row in events if not row["cache_hit"]]
    if [row["evaluation_count"] for row in fresh] != list(range(1, len(fresh) + 1)):
        raise AssertionError("evaluation sequence is inconsistent")
    if len({row["commitment_id"] for row in fresh}) != len(fresh) or len(fresh) > 40:
        raise AssertionError("unique candidate budget violated")
    if diagnostics["evaluation_count"] != len(fresh):
        raise AssertionError("evaluation counter is inconsistent")
    if any(row["cache_hit"] and row["improved"] for row in events):
        raise AssertionError("cache hit received improvement credit")
    if any(b["incumbent"] > a["incumbent"] + 1e-7 for a, b in zip(fresh, fresh[1:])):
        raise AssertionError("incumbent is not monotone")
    return validate_schedule(data, scenarios, result)


def validate_schedule(data, scenarios, result):
    """Validate a complete physical schedule independently of experiment budgets."""
    if not np.isfinite(result.objective):
        return {"feasible_solution": False, "trace_valid": True}
    if not commitment_is_feasible(data, scenarios, result.u, result.v, result.w):
        raise AssertionError("returned commitment is infeasible")
    residuals = {}
    fuel = 0.0
    next_shutdown = np.column_stack((result.w[:, 1:], np.zeros(data.n_units)))
    startup_cap = data.pmax[:, None] * result.u - (data.pmax - data.pstart)[:, None] * result.v
    shutdown_cap = data.pmax[:, None] * result.u - (data.pmax - data.pshut)[:, None] * next_shutdown
    if len(result.scenarios) != len(scenarios):
        raise AssertionError("missing scenario dispatch")
    for scenario, solution in zip(scenarios, result.scenarios):
        p, r = solution.p, solution.r
        if solution.name != scenario.name or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise AssertionError("invalid scenario dispatch")
        previous = np.column_stack((data.initial_output, p[:, :-1]))
        checks = {
            "balance": float(np.max(np.abs(p.sum(axis=0) - scenario.load))),
            "reserve": float(np.max(data.reserve_fraction * scenario.load - r.sum(axis=0))),
            "pmin": float(np.max(data.pmin[:, None] * result.u - p)),
            "startup_capacity": float(np.max(p + r - startup_cap)),
            "shutdown_capacity": float(np.max(p + r - shutdown_cap)),
            "ramp_up": float(np.max(p - previous - data.ramp_up[:, None])),
            "ramp_down": float(np.max(previous - p - data.ramp_down[:, None])),
            "nonnegative": float(max(np.max(-p), np.max(-r))),
        }
        residuals[scenario.name] = {key: max(0.0, value) for key, value in checks.items()}
        scenario_fuel = float(np.sum(data.cost_a[:, None] * p**2 + data.cost_b[:, None] * p))
        if abs(scenario_fuel - solution.objective) > 1e-4:
            raise AssertionError("scenario cost mismatch")
        fuel += scenario.probability * scenario_fuel
    maximum = max(value for row in residuals.values() for value in row.values())
    fixed = float(np.sum(data.cost_c[:, None] * result.u) + np.sum(data.startup_cost[:, None] * result.v)
                  + np.sum(data.shutdown_cost[:, None] * result.w))
    if maximum > 1e-5 or abs(fixed + fuel - result.objective) > 1e-4:
        raise AssertionError("dispatch or total cost validation failed")
    return {"feasible_solution": True, "trace_valid": True, "max_residual_mw": maximum,
            "scenario_residuals": residuals, "cost_residual": abs(fixed + fuel - result.objective)}


def prepare_manifest(directory, data_path, data, scenarios, llm_config, resume):
    source_paths = sorted((ROOT / "suc_benders").glob("*.py")) + [ROOT / "requirements.txt"]
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    settings = {
        "suite_version": 1, "seeds": list(SEEDS), "cases": list(CASES), "strategies": STRATEGIES,
        "max_iterations": 10, "max_candidate_evaluations": 40,
        "llm": asdict(llm_config), "gurobi_threads": 1, "bootstrap_samples": 10000,
        "bootstrap_seed": 20260930, "evaluation_checkpoints": [10, 20, 40],
        "time_checkpoints_seconds": [30, 60, 120, 300],
        "reference": asdict(FullConfig(time_limit=300, mip_gap=1e-4, output_flag=0)),
    }
    cases = {name: result_to_dict(case_scenarios(scenarios, name)) for name in CASES}
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "kaiwu", "gurobipy", "matplotlib")}
    thread_environment = {name: os.getenv(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    # Report generation can be repaired without re-running an expensive solver experiment.
    solver_sources = {name: value for name, value in source_hashes.items() if not name.endswith("attribution_report.py")}
    fingerprint = digest({"source": solver_sources, "data": result_to_dict(data), "cases": cases,
                          "settings": settings, "versions": versions, "thread_environment": thread_environment})
    path = directory / "manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("manifest already exists; use --resume to retain completed runs")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("solver code, inputs, packages or settings changed; use a new output directory")
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now(), "fingerprint": fingerprint, "settings": settings,
        "source_hashes": source_hashes, "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "cases": cases, "data": result_to_dict(data), "packages": versions,
        "python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "thread_environment": thread_environment,
        "schedule": schedule(), "research_runs": 150, "llm_logical_call_limit": 600,
        "llm_request_limit": 1200, "llm_remote_seed_supported": False,
    }
    for command, filename in [(["git", "rev-parse", "HEAD"], "commit.txt"),
                              (["git", "status", "--porcelain=v1"], "git-status.txt"),
                              (["git", "diff", "--binary", "HEAD"], "working-tree.patch")]:
        snapshot = subprocess.check_output(command, cwd=ROOT)
        (directory / filename).write_bytes(snapshot)
    for path in source_paths + sorted((ROOT / "tests").glob("*.py")) + [ROOT / "README.md", ROOT / "CHANGELOG.md"]:
        target = directory / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(path=directory / "inputs" / "units.json", value=json.loads(Path(data_path).read_text(encoding="utf-8")))
    for name, case in cases.items():
        write_json(directory / "inputs" / f"{name}.json", {"reserve_fraction": data.reserve_fraction, "scenarios": case})
    write_json(directory / "manifest.json", manifest)
    return manifest


def write_run(directory, payload):
    run_id = payload["run_id"]
    diagnostics = payload["result"].pop("diagnostics")
    for kind in ("evaluations", "proposals", "llm_calls", "sampling", "iterations"):
        path = directory / "traces" / f"{run_id}.{kind}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in diagnostics.pop(kind):
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    payload["diagnostics"] = diagnostics
    payload["complete"] = True
    write_json(directory / "runs" / f"{run_id}.json", payload)


def run_attribution(data_path, output_dir, *, resume=False, pilot=False, base_url=None, model=None):
    if not (os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip():
        raise ValueError("attribution requires an LLM credential in the process environment")
    directory = Path(output_dir).resolve()
    data, scenarios = load_suc_data(data_path)
    llm_config = LLMConfig(
        enabled=True, api_format="chat_completions", base_url=base_url, model=model,
        temperature=.2, max_tokens=8192, timeout_seconds=180, max_retries=2,
        retry_backoff_seconds=30, max_candidates=4, collect_diagnostics=True,
    )
    llm_config.base_url, llm_config.model = llm_settings(llm_config)
    manifest = prepare_manifest(directory, data_path, data, scenarios, llm_config, resume)
    gp = require_gurobi()
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    handler = logging.FileHandler(directory / "progress.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger("suc_benders")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        for number, row in enumerate(manifest["schedule"], 1):
            if pilot and number > 10:
                break
            path = directory / "runs" / f"{row['run_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if not saved.get("complete") or saved["fingerprint"] != manifest["fingerprint"]:
                    raise ValueError(f"invalid completed run: {row['run_id']}")
                continue
            active = {"state": "running", "active_run": row["run_id"], "ordinal": number, "total": 150, "updated_at": now()}
            write_json(directory / "progress.json", active)
            print(f"[{number}/150] START {row['run_id']}", flush=True)
            LOGGER.info("START %s", row["run_id"])
            current_scenarios = case_scenarios(scenarios, row["case"])
            config = research_config(row["strategy"], row["seed"], llm_config)
            started_at = now()
            try:
                result = solve_benders(data, current_scenarios, config)
                validation = validate_result(data, current_scenarios, result)
            except Exception as exc:
                write_json(directory / "errors" / f"{row['run_id']}.json", {
                    **row, "error_type": type(exc).__name__, "created_at": now(),
                })
                write_json(directory / "progress.json", {**active, "state": "error", "error_type": type(exc).__name__})
                raise
            calls = result.diagnostics["llm_calls"]
            request_count = sum(len(call.get("attempts", [])) for call in calls)
            if len(calls) > 10 or request_count > 20:
                raise AssertionError("LLM request budget violated")
            payload = {**row, "fingerprint": manifest["fingerprint"], "config": asdict(config),
                       "started_at": started_at, "finished_at": now(), "validation": validation,
                       "llm_calls": len(calls), "llm_requests": request_count,
                       "llm_failed_calls": sum(bool(call["failed"]) for call in calls),
                       "result": result_to_dict(result)}
            write_run(directory, payload)
            LOGGER.info("DONE %s objective=%.6f evaluations=%s runtime=%.2fs llm_calls=%s failed=%s",
                        row["run_id"], result.objective, result.diagnostics["evaluation_count"],
                        result.runtime, len(calls), payload["llm_failed_calls"])
            print(f"[{number}/150] DONE objective={result.objective:.6f} evaluations={result.diagnostics['evaluation_count']} "
                  f"runtime={result.runtime:.2f}s llm={len(calls)} failed={payload['llm_failed_calls']}", flush=True)
        completed = len(list((directory / "runs").glob("*.json")))
        if completed < 150:
            write_json(directory / "progress.json", {"state": "pilot_complete", "completed": completed, "total": 150, "updated_at": now()})
            print(f"Pilot complete: {completed}/150 runs. Resume with --resume.", flush=True)
            return
        # Reference solutions are deliberately unavailable until every research run is complete.
        for case in CASES:
            path = directory / "references" / f"{case}.json"
            if path.exists():
                continue
            write_json(directory / "progress.json", {"state": "reference", "case": case, "completed": 150, "updated_at": now()})
            config = FullConfig(time_limit=300, mip_gap=1e-4, output_flag=0)
            reference = solve_full(data, case_scenarios(scenarios, case), config)
            payload = {field.name: getattr(reference, field.name) for field in fields(reference) if field.name != "model"}
            write_json(path, {"case": case, "fingerprint": manifest["fingerprint"], "config": asdict(config),
                              "created_at": now(), **payload})
            reference.model.dispose()
            print(f"REFERENCE {case} objective={reference.objective:.6f} bound={reference.lower_bound:.6f} status={reference.status}", flush=True)
        from .attribution_report import generate_report
        generate_report(directory)
        write_json(directory / "progress.json", {"state": "complete", "completed": 150, "total": 150, "updated_at": now()})
        print(f"COMPLETE {directory / 'report.md'}", flush=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        handler.close()
