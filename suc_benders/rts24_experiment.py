"""One-shot, serial RTS-24 baseline validation; no LLM or automatic reruns."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .attribution import validate_schedule, write_json
from .data import load_suc_data
from .efficiency_experiment import read_events
from .full_model import FullConfig, solve_full
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders
from .subproblem import solve_scenario

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/rts24"
PROFILES = ("deterministic", "stochastic")
BUDGET = 600.0


def configuration():
    return BendersConfig(max_iterations=3, master_encoding="dynamic", decomposition="none",
        time_limit_seconds=BUDGET, decommit_candidates=False, random_candidates=False,
        collect_diagnostics=True, stop_on_no_new_candidates=False,
        quantum=QuantumConfig(kaiwu_seed=7, candidate_pool_size=16), llm=LLMConfig(enabled=False))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_object(row):
    return SimpleNamespace(objective=row["objective"],
        **{k: np.asarray(row[k]) for k in ("u", "v", "w")},
        scenarios=[SimpleNamespace(name=s["name"], objective=s.get("objective", s.get("fuel_cost")),
            p=np.asarray(s["p"]), r=np.asarray(s["r"])) for s in row["scenarios"]])


def worker(directory, profile, strategy):
    import gurobipy as gp
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 7)
    data, scenarios = load_suc_data(CASE / "units.json", CASE / f"{profile}.json")
    if strategy == "sa":
        with (directory / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
            def event(row):
                stream.write(json.dumps(result_to_dict(row), allow_nan=False) + "\n")
                stream.flush()
                if row["kind"] == "started":
                    write_json(directory / "ready.json", {"deadline": row["deadline_clock"]})
            config = configuration()
            config.event_sink = event
            write_json(directory / "result.json", solve_benders(data, scenarios, config))
    else:
        started = time.perf_counter()
        write_json(directory / "ready.json", {"started": started})
        result = solve_full(data, scenarios, FullConfig(time_limit=BUDGET, mip_gap=1e-4, output_flag=1))
        payload = {k: v for k, v in vars(result).items() if k != "model"}
        payload["solver_runtime_seconds"] = result.model.Runtime
        payload["wall_seconds"] = time.perf_counter() - started
        write_json(directory / "result.json", payload)
        # Independent continuous re-solves are validation, outside the reference budget.
        if np.isfinite(result.objective):
            checked = [solve_scenario(data, s, result.u, result.v, result.w,
                deadline=time.perf_counter() + 60, threads=1) for s in scenarios]
            recost = result.fixed_cost + sum(s.probability * r.objective for s, r in zip(scenarios, checked))
            write_json(directory / "fixed_dispatch_check.json", {
                "all_feasible": all(r.feasible for r in checked),
                "full_objective": result.objective, "recomputed_objective": recost,
                "difference": abs(recost - result.objective),
                "tolerance": 1e-3 + 1e-7 * abs(result.objective),
                "scenario_statuses": [r.status for r in checked],
                "result_sha256": hash_file(directory / "result.json"),
            })


def supervise(directory, profile, strategy):
    directory.mkdir(parents=True)
    write_json(directory / "state.json", {"status": "running"})
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "suc_benders.rts24_experiment",
            "--worker", strategy, "--profile", profile, "--output-dir", str(directory)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        reason = "completed"
        try:
            while process.poll() is None:
                ready = directory / "ready.json"
                elapsed = time.perf_counter() - start
                if not ready.exists() and elapsed >= 60:
                    reason = "startup_timeout"
                    break
                if ready.exists() and strategy == "sa" and time.perf_counter() >= read(ready)["deadline"]:
                    reason = "time_limit"
                    break
                # Gurobi has its own 600s solve limit; allow modeling, serialization
                # and at most three 60s fixed-dispatch validation solves afterwards.
                if elapsed >= 900:
                    reason = "worker_timeout"
                    break
                time.sleep(.2)
        except KeyboardInterrupt:
            reason = "interrupted"
            raise
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            if reason == "completed" and process.returncode != 0:
                reason = "failed"
            write_json(directory / "state.json", {"status": reason, "exitcode": process.returncode,
                "wall_seconds": time.perf_counter() - start})
    print(f"{profile}/{strategy}: {reason}", flush=True)


def audit(directory):
    manifest = read(directory / "manifest.json")
    for name, digest in manifest["source_sha256"].items():
        if hash_file(ROOT / name) != digest:
            raise ValueError(f"audit source mismatch: {name}")
    report = {"passed": True, "profiles": {}, "protocol": manifest["protocol"]}
    for profile in PROFILES:
        data, scenarios = load_suc_data(CASE / "units.json", CASE / f"{profile}.json")
        sa_dir, ref_dir = directory / profile / "sa", directory / profile / "reference"
        events = read_events(sa_dir / "events.jsonl")
        candidates = [e for e in events if e["kind"] == "evaluation" and e["feasible"]
                      and e["objective"] is not None and e["elapsed_seconds"] <= BUDGET]
        checks = [validate_schedule(data, scenarios, schedule_object(e)) for e in candidates]
        best = min(candidates, key=lambda e: e["objective"]) if candidates else None
        ref = read(ref_dir / "result.json") if (ref_dir / "result.json").exists() else None
        ref_check = validate_schedule(data, scenarios, schedule_object(ref)) if ref and ref["objective"] is not None else None
        consistency = read(ref_dir / "fixed_dispatch_check.json") if (ref_dir / "fixed_dispatch_check.json").exists() else None
        consistent = bool(consistency and consistency["all_feasible"] and consistency["difference"] is not None
                          and consistency["difference"] <= consistency["tolerance"]
                          and consistency["result_sha256"] == hash_file(ref_dir / "result.json"))
        completed_sa = sum(e["kind"] == "sa_end" and e["status"] == "completed" and e["elapsed_seconds"] <= BUDGET for e in events)
        completed_rounds = sum(e["kind"] == "iteration" and e["elapsed_seconds"] <= BUDGET for e in events)
        sa_result = read(sa_dir / "result.json") if (sa_dir / "result.json").exists() else None
        # Diagnostics verify that returned samples went through decoding and repair.
        processed = bool(sa_result and any(p.get("origin") == "sa" for p in sa_result["diagnostics"]["proposals"]))
        states = {"sa": read(sa_dir / "state.json"), "reference": read(ref_dir / "state.json")}
        passed = bool(best and ref_check and consistent and completed_sa and processed
                      and states["sa"]["status"] in ("completed", "time_limit")
                      and states["reference"]["status"] == "completed"
                      and not any(e["kind"] == "llm_start" for e in events))
        bound = ref.get("lower_bound") if ref else None
        if best and bound is not None and bound > best["objective"] + 1e-3:
            raise AssertionError("reference lower bound exceeds validated SA cost")
        row = {"passed": passed, "states": states, "sa_completed": completed_sa,
            "completed_rounds": completed_rounds, "samples_processed": processed,
            "validated_candidates": len(checks), "sa_objective": best["objective"] if best else None,
            "reference_objective": ref["objective"] if ref else None, "reference_lower_bound": bound,
            "reference_status": ref["status"] if ref else None, "reference_mip_gap": ref["mip_gap"] if ref else None,
            "max_residual_mw": max((c["max_residual_mw"] for c in checks + ([ref_check] if ref_check else [])), default=None),
            "fixed_dispatch_check": consistency,
            "relative_reference_cost_difference": best["objective"] / ref["objective"] - 1 if best and ref and ref["objective"] else None,
            "certified_relative_error_bound": (best["objective"] - bound) / bound if best and bound and bound > 0 else None,
            "sa_reliable_lower": sa_result["diagnostics"].get("reliable_lower") if sa_result else None}
        report["profiles"][profile] = row
        report["passed"] &= passed
    write_json(directory / "audit.json", report)
    lines = ["# IEEE RTS-24 派生无网络 UC/SUC：基线验证", "",
        "32 台机组、24 小时。固定启停费用；无网络、冷热启动或水电能量约束。",
        "三场景倍率、概率及 5% 备用是项目设定，不是 IEEE 原始数据。", "",
        f"验收：{'通过' if report['passed'] else '未通过'}。运行源码提交：`{manifest['commit']}`。", "",
        "|配置|SA 完整轮次/调用|SA 可行成本|Gurobi 可行成本|参考成本差|认证误差上界|",
        "|---|---:|---:|---:|---:|---:|"]
    def number(value, percent=False):
        return "未获得" if value is None else (f"{100*value:.6f}%" if percent else f"{value:.6f}")
    for name, row in report["profiles"].items():
        lines.append(f"|{name}|{row['completed_rounds']}/{row['sa_completed']}|{number(row['sa_objective'])}|"
                     f"{number(row['reference_objective'])}|{number(row['relative_reference_cost_difference'], True)}|"
                     f"{number(row['certified_relative_error_bound'], True)}|")
    lines += ["", "参考成本差 = SA成本 / Gurobi可行成本 − 1；认证误差上界 = (SA成本 − Gurobi下界) / Gurobi下界。",
        "认证依赖 Gurobi 的数值容差与下界，不是 SA 独立证明。完整下界、状态、MIP gap、残差与成本复核见 audit.json。", "",
        "每个配置仅一次 SA（种子7、动态整时域、候选池16、最多3轮/600秒）及一次独立 Gurobi参考",
        "（单线程、种子7、求解时限600秒、MIPGap=1e-4）。先完成全部 SA，再求参考，不反馈参考解。",
        "SA保留确定性初始化与修复，关闭LLM、独立减开机和随机候选；无量子真机。",
        "SA使用端到端截止时间；Gurobi的600秒是求解时限，建模与输出开销另计。固定方案连续重求解仅用于验收。",
        "这是单种子的接入验证，不作速度、精度优势或替代求解器结论；没有为改善成绩追加运行。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--worker", choices=("sa", "reference"), help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if args.worker:
        worker(directory, args.profile, args.worker)
        return
    if args.audit_only:
        if not audit(directory)["passed"]:
            raise SystemExit("baseline acceptance failed")
        return
    if directory.exists():
        raise ValueError("output directory must be new; use --audit-only for saved runs")
    subprocess.run([sys.executable, str(CASE / "convert.py"), "--check"], check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("commit source changes before running the baseline")
    paths = sorted((ROOT / "suc_benders").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    paths += [p for p in CASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    hashes = {p.relative_to(ROOT).as_posix(): hash_file(p) for p in paths}
    protocol = {"sa": asdict(configuration()), "reference": asdict(FullConfig(time_limit=BUDGET, mip_gap=1e-4)),
                "reference_threads": 1, "reference_seed": 7, "order": [f"{p}/{s}" for s in ("sa", "reference") for p in PROFILES]}
    write_json(directory / "manifest.json", {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": hashes, "protocol": protocol, "python": sys.version, "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "gurobipy", "kaiwu")}})
    for strategy in ("sa", "reference"):
        for profile in PROFILES:
            supervise(directory / profile / strategy, profile, strategy)
    if not audit(directory)["passed"]:
        raise SystemExit("baseline acceptance failed; keep evidence and do not publish a release")


if __name__ == "__main__":
    main()
