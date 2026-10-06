"""Fixed three-unit, 24-hour convergence comparison for 1/3/5-hour blocks."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .block_master_model import priced_blocks
from .paper_model import IndexedUC
from .sa_scale_experiment import (ROOT, SEEDS, case_input, checksum, config,
    end_statistics, events, load_case, read, save, supervise)
from .subagent import sha
from .validation import schedule_object, validate_schedule


HOURS = (1, 3, 5)
CASE = "n3-det"
RUN_SECONDS = 90
TOTAL_SECONDS = 1200


def definitions():
    return [dict(id=f"n3-det-h{hours}-varying-{seed}", case=CASE,
                 hours=hours, seed=seed, mode="varying")
            for hours in HOURS for seed in SEEDS]


def convergence_trace(directory, definition):
    folder = directory / "end_to_end" / definition["id"]
    rows = events(folder / "events.jsonl")
    inner = [dict(r, gap=r["upper"]-r["lower"]) for r in rows if r["kind"] == "inner_iteration"]
    outer = [{k: r[k] for k in ("iteration", "elapsed_seconds", "incumbent", "reliable_lower", "coordination_upper")}
             for r in rows if r["kind"] == "iteration"]
    blocks = {(r["outer"], r["block"]): dict(outer=r["outer"], block=r["block"],
        times=list(range(r["block"]*definition["hours"], min((r["block"]+1)*definition["hours"], 24))),
        lower=r["lower"], upper=r["upper"], gap=r["gap"], cuts=r["cuts"],
        inner_iterations=r["inner"], converged=r["converged"]) for r in inner}
    for r in read(folder / "result.json").get("diagnostics", {}).get("blocks", []):
        if (r["outer"], r["block"]) not in blocks:
            blocks[r["outer"], r["block"]] = dict(outer=r["outer"], block=r["block"], times=r["times"],
                lower=r["lower"], upper=r["upper"], gap=r["upper"]-r["lower"] if r["upper"] is not None and r["lower"] is not None else None,
                cuts=None, inner_iterations=r["inner_iterations"], converged=r["converged"])
    trace = dict(inner=inner, outer=outer, blocks=list(blocks.values()))
    save(folder / "convergence.json", trace)
    return trace


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def plot_traces(directory, runs, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    for kind, filename in (("inner", "inner-first-block.png"), ("outer", "outer-convergence.png")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=kind == "outer")
        for axis, hours in zip(axes, HOURS):
            for index, run in enumerate(r for r in runs if r["hours"] == hours):
                rows = traces[run["id"]][kind]
                if kind == "inner":
                    rows = [r for r in rows if r["outer"] == 1 and r["block"] == 0]
                    x, lo, hi = "inner", "lower", "upper"
                else:
                    x, lo, hi = "iteration", "reliable_lower", "coordination_upper"
                for field, style, label in ((lo, "--", "LB"), (hi, "-", "UB")):
                    points = [(r[x], r[field]) for r in rows if r.get(field) is not None]
                    if points:
                        axis.plot(*zip(*points), style, color=f"C{index}", linewidth=1.6,
                                  marker="o" if kind == "outer" else None, markersize=4,
                                  label=f"{run['seed']} {label}")
            if not axis.lines:
                axis.text(.5, .5, "No completed outer rounds" if kind == "outer" else "No inner records",
                          ha="center", va="center", transform=axis.transAxes)
            else:
                axis.legend(fontsize=8)
            if kind == "outer":
                completed = sorted({r["iteration"] for run in runs if run["hours"] == hours
                                    for r in traces[run["id"]]["outer"]})
                axis.set_xticks(completed)
                if completed:
                    axis.set_xlim(.5, max(completed)+.5)
                missing = [str(r["seed"]) for r in runs if r["hours"] == hours and not traces[r["id"]]["outer"]]
                if missing and axis.lines:
                    axis.text(.03, .04, ", ".join(missing)+": no completed outer round", fontsize=8,
                              transform=axis.transAxes)
            axis.set_title(f"{hours}-hour blocks")
            axis.set_xlabel("Inner iteration (outer 1, block 1)" if kind == "inner" else "Completed outer round")
            axis.set_ylabel("Local priced objective" if kind == "inner" else "Lagrangian dual bounds")
            if kind == "inner":
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


def report(directory):
    manifest = read(directory / "manifest.json")
    for path, digest in manifest["source_sha256"].items():
        if sha(directory / "source" / path) != digest:
            raise ValueError(f"archived source mismatch: {path}")
        if path != "suc_benders/three_unit_block_experiment.py" and sha(ROOT / path) != digest:
            raise ValueError(f"solver/input source changed: {path}")
    for path, digest in manifest["input_sha256"].items():
        if sha(directory / path) != digest:
            raise ValueError(f"experiment input changed: {path}")
    data, scenarios = load_case(directory, CASE)
    reference = read(directory / "references" / CASE / "result.json")
    if reference.get("objective") is not None:
        reference["validation"] = validate_schedule(data, scenarios, schedule_object(reference))
    runs, traces = [], {}
    for definition in manifest["runs"]:
        row = end_statistics(directory, definition)
        trace = convergence_trace(directory, definition)
        traces[row["id"]] = trace
        failed = next((b for b in trace["blocks"] if not b["converged"]), None)
        row.update(completed_outer_rounds=len(trace["outer"]), first_unconverged_block=failed,
            coordination_upper=read(directory / "end_to_end" / row["id"] / "result.json").get("diagnostics", {}).get("coordination_upper"),
            dual_converged=row["solver_status"] == "DUAL_CONVERGED",
            reached_reference_optimum=bool(reference.get("certified") and row["final_cost"] is not None
                and row["validated_schedules"] > 0 and abs(row["final_cost"]-reference["objective"]) <= 1e-4))
        row["raw_discrete_feasible_rate"] = row["raw_discrete_feasible"]/row["raw_samples"] if row["raw_samples"] else None
        runs.append(row)
    groups = []
    for hours in HOURS:
        selected = [r for r in runs if r["hours"] == hours]
        costs = [r["final_cost"] for r in selected if r["final_cost"] is not None]
        seconds = [r["state"]["wall_seconds"] for r in selected if "wall_seconds" in r["state"]]
        groups.append(dict(hours=hours, dual_converged_seeds=sum(r["dual_converged"] for r in selected),
            optimal_seeds=sum(r["reached_reference_optimum"] for r in selected),
            median_cost=statistics.median(costs) if costs else None, worst_cost=max(costs) if costs else None,
            median_seconds=statistics.median(seconds) if seconds else None, worst_seconds=max(seconds) if seconds else None))
    summary = dict(execution_commit=manifest["commit"], execution_dirty_paths=manifest["dirty_paths"],
        report_generator_sha256=sha(Path(__file__)), reference=reference,
        reference_state=read(directory / "references" / CASE / "state.json"),
        end_to_end=runs, groups=groups, baseline_reproduced=groups[0]["optimal_seeds"] == 3,
        warning="Three-seed screening; time/iteration limits retained. SA improvement includes full-day recovery.")
    generator = directory / "report-source/suc_benders/three_unit_block_experiment.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_bytes(Path(__file__).read_bytes())
    save(directory / "summary.json", summary)
    accounted = len(runs) == 9 and all(r["state"].get("status") in
        ("completed", "time_limit", "failed", "phase_budget_exhausted") for r in runs)
    save(directory / "audit.json", dict(passed=accounted,
        all_nine_slots_recorded=accounted, all_returned_schedules_valid=True,
        full_results_saved=sum(not r["partial"] for r in runs),
        source_and_input_hashes_valid=True, block_lengths=manifest["block_lengths"],
        max_residual_mw=max((r["max_residual_mw"] for r in runs if r["max_residual_mw"] is not None), default=None),
        baseline_reproduced=summary["baseline_reproduced"]))
    write_csv(directory / "runs.csv", runs)
    write_csv(directory / "groups.csv", groups)
    write_csv(directory / "blocks.csv", [dict(run_id=r["id"], hours=r["hours"], seed=r["seed"], **b)
              for r in runs for b in traces[r["id"]]["blocks"]])
    plot_traces(directory, runs, traces)
    def number(v):
        return "NA" if v is None else f"{v:,.2f}"
    def pct(v):
        return "NA" if v is None else f"{100*v:.4f}%"
    lines = ["# 固定三机组：1／3／5 小时分块收敛实验", "",
        "固定 RTS 机组 3、9、20；24 小时，单场景，零备用；每次进程预算 90 秒，内层 60 次、外层 20 次。",
        "三个基础种子逐次变化；不调参、不追加运行。5 小时分块为 5+5+5+5+4。", "",
        f"执行基于 `{manifest['commit']}`；具体执行源码及未提交路径记录在 manifest.json，源码副本和 SHA256 均保存。",
        f"本轮 Gurobi 参考：{number(reference.get('objective'))}；已认证：{reference.get('certified', False)}。", "",
        "| 分块 | 种子 | 状态 | 最终成本 | SA加恢复降本 | 高于参考 | 原问题gap | 论文口径gap | 首轮完成块 | 外层完成轮 | 乘子更新 | 秒 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['hours']}h | {r['seed']} | {r['solver_status'] or r['state'].get('status')} | {number(r['final_cost'])} | {number(r['sa_with_recovery_improvement'])} | {pct(r['relative_reference_cost_difference'])} | {pct(r['primal_gap'])} | {pct(r['paper_style_gap'])} | {r['first_outer_blocks_converged']}/{r['first_outer_blocks_total']} | {r['completed_outer_rounds']} | {r['multiplier_updates']} | {number(r['state'].get('wall_seconds'))} |")
    lines += ["", "| 分块 | 对偶收敛 | 达到参考最优 | 成本中位数 | 最差成本 | 耗时中位数 | 最长耗时 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['hours']}h | {g['dual_converged_seeds']}/3 | {g['optimal_seeds']}/3 | {number(g['median_cost'])} | {number(g['worst_cost'])} | {number(g['median_seconds'])} | {number(g['worst_seconds'])} |")
    lines += ["", f"1 小时基线复现：{summary['baseline_reproduced']}。所有返回的方案均通过独立物理及成本校验。",
        "DUAL_CONVERGED 仅表示对偶收敛；达到原问题最优还要求物理校验及与已认证参考目标一致。",
        "三种 gap 分开报告。参考缺失为 NA；未完成外层不补零；不同块和不同乘子下的内层界不连接。",
        "三个种子仅用于初筛；时间或迭代上限不能解释为方法永远不收敛。成本改进属于 SA 加全日恢复流程。", "",
        "![首轮首块的内层上下界](inner-first-block.png)", "",
        "![已完成外层的上下界](outer-convergence.png)", "",
        "逐块记录见 blocks.csv；逐种子指标见 runs.csv；完整事件及方案在 end_to_end/。",
        "重建：python -m suc_benders.three_unit_block_experiment --report-only --output-dir <本目录>"]
    (directory / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run(directory):
    if directory.exists():
        raise ValueError("output directory already exists; use a new directory")
    directory.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started+TOTAL_SECONDS
    source = list((ROOT / "suc_benders").glob("*.py"))+list((ROOT / "data/rts24").glob("*.json"))+[ROOT / "requirements.txt"]
    manifest = dict(protocol="three-unit-blocks", runs=definitions(), snapshots=[], trials=[],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty_paths=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        seeds=SEEDS, unit_rows=[3, 9, 20], horizon=24, hours=HOURS,
        run_seconds=RUN_SECONDS, reference_seconds=30, total_seconds=TOTAL_SECONDS,
        started_unix=time.time(), started_clock=started, global_deadline=deadline,
        python=sys.version, dependencies={k: importlib.metadata.version(k) for k in ("numpy", "gurobipy", "kaiwu", "matplotlib")},
        configs={d["id"]: asdict(config(d["seed"], "varying", RUN_SECONDS, d["hours"])) for d in definitions()},
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source})
    for path in manifest["source_sha256"]:
        target = directory / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    save(directory / "inputs" / CASE / "units.json", case_input(3))
    save(directory / "inputs" / CASE / "scenarios.json", read(ROOT / "data/rts24/deterministic.json"))
    manifest["input_sha256"] = {p.relative_to(directory).as_posix(): sha(p) for p in (directory / "inputs").rglob("*.json")}
    data, scenarios = load_case(directory, CASE)
    uc = IndexedUC(data, scenarios)
    manifest["block_lengths"] = {}
    for hours in HOURS:
        blocks, cross = priced_blocks(uc, hours, [0.]*len(uc.rows))
        if [t for b in blocks for t in b.times] != list(range(24)) or sorted([i for b in blocks for i in b.row_ids]+cross) != list(range(len(uc.rows))):
            raise ValueError("time/constraint partition incomplete")
        manifest["block_lengths"][str(hours)] = [len(b.times) for b in blocks]
    save(directory / "manifest.json", manifest)
    supervise(directory, directory / "references" / CASE, "reference", CASE, 30, deadline)
    for definition in manifest["runs"]:
        save(directory / "progress.json", dict(run=definition["id"], started=time.perf_counter()))
        supervise(directory, directory / "end_to_end" / definition["id"], "end", definition["id"], RUN_SECONDS, deadline-120)
        print(f"finished {definition['id']}", flush=True)
    supervise(directory, directory / "report-worker", "report", "all", 120, deadline)
    save(directory / "state.json", dict(status="completed" if read(directory / "report-worker/state.json").get("status") == "completed" and read(directory / "audit.json").get("passed") else "report_incomplete",
        seconds=time.perf_counter()-started, within_total_budget=time.perf_counter() <= deadline))
    checksum(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.report_only:
        report(directory)
        checksum(directory)
    else:
        run(directory)


if __name__ == "__main__":
    main()
