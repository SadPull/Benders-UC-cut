"""Run with python -m suc_benders.exact_experiment; independent of SA runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from .attribution import digest, now, write_json
from .data import load_suc_data
from .exact_control import Budget, aggregate, closed_loop, solve_block
from .paper_model import Block, IndexedUC
from .solver import result_to_dict

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("D_block1", 1), ("E_block2", 2), ("F_block3", 3))
SEEDS = (7, 17, 27, 37, 47)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source, data, scenarios):
    manifest = read(source/"manifest.json")
    # These exact files define the archived model, ordering, objective and dispatch.
    for name in ("paper_model", "paper_solver", "data", "common", "subproblem", "candidates", "master", "solver"):
        relative = f"suc_benders/{name}.py"
        if file_hash(ROOT/relative) != manifest["sources"][relative]:
            raise ValueError(f"source model mismatch: {relative}")
    if digest(result_to_dict(asdict(data))) != digest(manifest["data"]):
        raise ValueError("source input mismatch")
    if digest(result_to_dict([asdict(s) for s in scenarios])) != digest(manifest["scenarios"]):
        raise ValueError("source scenarios mismatch")
    return manifest


def snapshots(source, data, scenarios):
    runs = {(method, seed): read(source/"runs"/f"{method}__s{seed}.json")
            for method, _ in METHODS for seed in SEEDS}
    models = {h: IndexedUC(data, scenarios, h) for _, h in METHODS}
    jobs, seen = [], set()
    for phase in range(3):
        for seed in SEEDS:
            for method, hours in METHODS:
                run = runs[method, seed]
                rows = run["result"]["diagnostics"]["outer"]
                if not rows:
                    continue
                row = rows[(0, (len(rows)-1)//2, len(rows)-1)[phase]]
                ident = f"{method}__s{seed}__o{row['outer']}"
                if ident in seen:
                    continue
                seen.add(ident)
                prices = row["prices"]
                if len(prices) != len(models[hours].coupling) or not np.all(np.isfinite(prices)):
                    raise ValueError(f"invalid multiplier vector: {ident}")
                jobs.append(dict(id=ident, hours=hours, seed=seed, method=method, phase=phase,
                                 outer=row["outer"], prices=prices,
                                 sampled=row["sampled_lagrangian_value"],
                                 historical_lower=row["reliable_lower"],
                                 final_historical_lower=run["result"]["diagnostics"]["reliable_lower"],
                                 cache_key=digest([hours, prices])))
    return jobs


def offline(data, scenarios, jobs, state, budget, block_limit):
    state.setdefault("cache", {})
    state.setdefault("snapshots", [])
    done = {r["id"] for r in state["snapshots"]}
    for job in jobs:
        if job["id"] in done:
            continue
        cached = state["cache"].get(job["cache_key"])
        if cached is None and budget.available():
            uc = IndexedUC(data, scenarios, job["hours"])
            costs, constant = uc.coefficients(np.asarray(job["prices"]))
            pending = state.setdefault("pending", {"key": job["cache_key"], "records": []})
            if pending["key"] != job["cache_key"]:
                raise ValueError("pending snapshot mismatch")
            for j in range(len(pending["records"]), len(uc.blocks)):
                if not budget.available():
                    break
                pending["records"].append(solve_block(Block(uc, j), costs, budget.deadline, block_limit))
                budget.save()
            cached = {"blocks": pending["records"], **aggregate(pending["records"], len(uc.blocks), constant)}
            cached["complete"] = len(pending["records"]) == len(uc.blocks)
            state["cache"][job["cache_key"]] = cached
            del state["pending"]
        if cached is None:
            row = {**job, "status": "BUDGET_EXHAUSTED", "lower": None, "upper": None, "certified": False}
        else:
            row = {**job, **{k: v for k, v in cached.items() if k != "blocks"},
                   "status": "COMPLETE" if cached["complete"] else "INCOMPLETE"}
            row["sample_error_lower"] = job["sampled"]-cached["upper"] if cached["upper"] is not None else None
            row["sample_error_upper"] = job["sampled"]-cached["lower"] if cached["lower"] is not None else None
            # A prior SA incumbent can beat a time-limited control incumbent, but not its bound.
            if row["sample_error_upper"] is not None and row["sample_error_upper"] < -1e-3:
                raise AssertionError("source sampled value below exact valid lower bound")
        state["snapshots"].append(row)
        budget.save()
        print(f"{now()} offline {job['id']} {row['status']}", flush=True)
    state["finished"] = True
    budget.save(finished=True)


def checkpoint(state, seconds):
    objectives = [e["objective"] for e in state.get("evaluations", []) if e["elapsed"] <= seconds]
    bounds = [r["best_lower"] for r in state.get("rounds", [])
              if r["bound_elapsed"] <= seconds and r["best_lower"] is not None]
    return dict(seconds=seconds, objective=min(objectives) if objectives else None,
                lower=max(bounds) if bounds else None)


def report(output, source):
    reference = read(source/"reference.json")
    z = reference["objective"]
    gap = lambda lb: None if lb is None else (z-lb)/z*100
    fmt = lambda v: "—" if v is None else f"{v:,.4f}"
    lines = ["# 有限预算精确块对照", "", f"Gurobi 参考目标：{z:,.6f}；参考下界：{reference['lower_bound']:,.6f}。",
             "下界 gap = (参考目标 − 可靠下界) / 参考目标；可行解差距 = (可行成本 − 参考目标) / 参考目标。",
             "精确指给定容差内认证；超时值仅构成上下界区间。", ""]
    summary = {"reference": {k: reference[k] for k in ("objective", "lower_bound", "mip_gap")}}
    path = output/"offline.json"
    if path.exists():
        state = read(path)
        rows = state.get("snapshots", [])
        records = [r for item in state.get("cache", {}).values() for r in item["blocks"]]
        summary["offline"] = dict(snapshots=len(rows), complete=sum(r["status"] == "COMPLETE" for r in rows),
                                  certified_snapshots=sum(r["certified"] for r in rows),
                                  unique_blocks=len(records), certified_blocks=sum(r["certified"] for r in records),
                                  used_seconds=state["budget"]["used"])
        lines += ["## 离线同乘子诊断", "", str(summary["offline"]), "",
                  "| 分块 | 有下界快照数 | 最好精确下界 gap (%) | 原最终下界最好 gap (%) | 采样误差区间均值 |",
                  "|---|---:|---:|---:|---:|"]
        for method, _ in METHODS:
            group = [r for r in rows if r["method"] == method]
            lower = [r["lower"] for r in group if r["lower"] is not None]
            errors = [r for r in group if r.get("sample_error_lower") is not None and r.get("sample_error_upper") is not None]
            mean_lo = float(np.mean([r["sample_error_lower"] for r in errors])) if errors else None
            mean_hi = float(np.mean([r["sample_error_upper"] for r in errors])) if errors else None
            old = [r["final_historical_lower"] for r in group]
            lines.append(f"| {method} | {len(lower)} | {fmt(gap(max(lower)) if lower else None)} | {fmt(gap(max(old)) if old else None)} | [{fmt(mean_lo)}, {fmt(mean_hi)}] |")
        lines += ["", "历史最好下界可能来自其他乘子，不能作为当前乘子的连续松弛值。误差单位为原成本单位；共享快照保留各运行映射。", ""]
    lines += ["## 独立闭环", "", "| 分块 | 状态 | 已完成轮 | 300秒下界 gap (%) | 300秒可行成本 | 可行成本差距 (%) |",
              "|---|---|---:|---:|---:|---:|"]
    summary["closed"] = {}
    for method, hours in METHODS:
        path = output/f"closed_{hours}.json"
        if not path.exists():
            continue
        state = read(path)
        checks = [checkpoint(state, t) for t in (30, 60, 120, 300)]
        result = checks[-1]
        cost_gap = None if result["objective"] is None else (result["objective"]-z)/z*100
        summary["closed"][method] = dict(status=state.get("status"), checkpoints=checks, used_seconds=state["budget"]["used"])
        lines.append(f"| {method} | {state.get('status')} | {len(state.get('rounds', []))} | {fmt(gap(result['lower']))} | {fmt(result['objective'])} | {fmt(cost_gap)} |")
        old = read(source/"runs"/f"{method}__s7.json")["result"]
        events = old["diagnostics"]["evaluations"]
        for check in checks:
            values = [e["objective"] for e in events if e.get("feasible") and e["elapsed_seconds"] <= check["seconds"]]
            check["sa_seed7_objective"] = min(values) if values else None
            check["objective_difference_vs_sa"] = (check["objective"]-min(values)
                                                    if values and check["objective"] is not None else None)
    lines += ["", "## 解读边界", "", "- 离线同乘子区间用于判断局部近似误差；更紧下界与旧下界的差异也包含下界构造方式变化。",
              "- 闭环是单候选精确块策略与多候选 SA 策略的比较；不能把差异全归因于求解器。",
              "- 乘子仍限制在原盒内；10轮或时间结束不意味着对偶收敛，盒内收敛也不代表原问题最优。",
              "- 所有完整候选均经现有物理和成本验证。检查点只采用截止前完成结果，原始日志保留超时完成记录。",
              "- 未完成/未认证结果不作最优性结论；没有真机、QUBO、SA或LLM调用。",
              "- 30/60/120/300秒与旧SA种子7的配对成本见 summary.json；原实验其余种子用于离线诊断。", ""]
    write_json(output/"summary.json", summary)
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT/"results/paper-transfer-20260930-v2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("both", "offline", "closed"), default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir or ROOT/"results"/f"exact-block-control-{now().replace(':', '').replace('+', '_')}"
    data, scenarios = load_suc_data(ROOT/"data/suc_10x24.json")
    source_manifest = verify_source(source, data, scenarios)
    jobs = snapshots(source, data, scenarios)
    identity = dict(source_fingerprint=source_manifest["fingerprint"],
                    source_files={str(p.relative_to(source)): file_hash(p) for p in
                                  [source/"manifest.json", source/"reference.json"] + sorted((source/"runs").glob("[DEF]*.json"))},
                    sources={p.name: file_hash(p) for p in sorted((ROOT/"suc_benders").glob("*.py"))},
                    packages={p: importlib.metadata.version(p) for p in ("numpy", "gurobipy", "kaiwu")},
                    python=sys.version, mode=args.mode, block_seconds=30., offline_seconds=1800.,
                    closed_seconds=300., outer_rounds=10, threads=1, seed=7, mip_gap=1e-8, mip_gap_abs=1e-4)
    fingerprint = digest(identity)
    manifest_path = output/"manifest.json"
    if output.exists():
        if not args.resume or not manifest_path.exists() or read(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError("existing output requires --resume and identical inputs, source and settings")
    else:
        if args.resume:
            raise ValueError("resume output does not exist")
        output.mkdir(parents=True)
        write_json(manifest_path, dict(created_at=now(), fingerprint=fingerprint, **identity,
                                      data=asdict(data), scenarios=[asdict(s) for s in scenarios],
                                      interrupted_budget_policy="charge wall time since last active checkpoint, including downtime"))
        for p in sorted((ROOT/"suc_benders").glob("*.py")):
            (output/"source").mkdir(exist_ok=True)
            shutil.copy2(p, output/"source"/p.name)
    print(f"{now()} output {output}", flush=True)
    if args.mode in ("both", "offline"):
        path = output/"offline.json"
        state = read(path) if path.exists() else {}
        if not state.get("finished"):
            budget = Budget(path, state, 1800.)
            offline(data, scenarios, jobs, state, budget, 30.)
    if args.mode in ("both", "closed"):
        for _, hours in METHODS:
            path = output/f"closed_{hours}.json"
            state = read(path) if path.exists() else {}
            if not state.get("finished"):
                print(f"{now()} closed {hours} start", flush=True)
                budget = Budget(path, state, 300.)
                closed_loop(data, scenarios, hours, state, budget)
                print(f"{now()} closed {hours} {state['status']}", flush=True)
    report(output, source)
    print(f"{now()} report {output/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
