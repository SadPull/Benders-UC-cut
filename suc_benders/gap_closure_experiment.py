"""Frozen, serial RTS-24 gap-closure protocol with a 7200-second envelope."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .master import QuantumConfig
from .sa_scale_experiment import SEEDS, append, case_input, checksum, events, load_case, read, save
from .solver import BendersConfig, result_to_dict
from .subagent import sha
from .validation import schedule_object, validate_schedule


ROOT = Path(__file__).resolve().parents[1]
HOURS = (1, 2, 3, 5)
PHASE_SECONDS = dict(reference=720, sanity=360, deterministic=240, screening=1200,
                     native_confirmation=600, comparison=2100, confirmation=1800, audit=180)
TOTAL_SECONDS = sum(PHASE_SECONDS.values())


def definitions():
    runs = []
    def add(stage, case, hours, method, seed, seconds):
        runs.append(dict(id=f"{stage}-{case}-h{hours or 'selected'}-{method}-{seed}",
                         stage=stage, case=case, hours=hours, method=method, seed=seed, seconds=seconds))
    for method in ("sa_only", "sa_fallback"):
        for seed in SEEDS:
            add("sanity", "n3-det", 1, method, seed, 60)
    for case in ("n10-det", "n32-det"):
        add("deterministic", case, 1, "native_only", 7, 120)
    for hours in HOURS:
        add("screening", "n32-stoch", hours, "native_only", 7, 300)
    add("native_confirmation", "n32-stoch", None, "native_only", 7, 600)
    for method in ("sa_only", "sa_fallback", "native_only"):
        for seed in (SEEDS if method != "native_only" else (7,)):
            add("comparison", "n32-stoch", None, method, seed, 300)
    for seed in SEEDS:
        add("confirmation", "n32-stoch", None, "sa_fallback", seed, 600)
    return runs


def configuration(definition, seconds=None):
    method = definition["method"]
    sanity = definition["stage"] == "sanity"
    return BendersConfig(max_iterations=20 if sanity else 40,
        max_inner_iterations=60 if sanity else 200, block_hours=definition["hours"],
        convergence_tolerance=1e-4, quantum_candidates=method != "native_only",
        native_fallback_patience={"sa_only": None, "native_only": 1, "sa_fallback": 3}[method],
        defer_full_recovery=not (sanity and method == "sa_only"),
        sa_seed_per_call=True, quantum=QuantumConfig(kaiwu_seed=definition["seed"]),
        time_limit_seconds=seconds if seconds is not None else definition["seconds"])


def update_checkpoint(checkpoint, row, deadline):
    """Only fully emitted, on-time evaluations and certified bounds can score."""
    if row["clock"] > deadline:
        return False
    if row["kind"] == "evaluation" and row.get("feasible") and row.get("improved"):
        checkpoint["best"] = result_to_dict(row)
    elif row["kind"] == "bound_update":
        value = row.get("reliable_lower")
        if value is None or not math.isfinite(value):
            return False
        checkpoint["bound"] = result_to_dict(row)
        if row.get("coordination_upper") is not None and math.isfinite(row["coordination_upper"]):
            checkpoint["coordination"] = result_to_dict(row)
    else:
        return False
    checkpoint["clock"] = row["clock"]
    return True


def end_worker(directory, definition, output, deadline):
    from .solver import solve_benders
    # This worker intentionally never opens the reference directory.
    data, scenarios = load_case(directory, definition["case"])
    cfg = configuration(definition, max(.001, deadline-time.perf_counter()-2.))
    save(output / "config.json", asdict(cfg))
    checkpoint = {}
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        def sink(row):
            append(stream, row)
            if update_checkpoint(checkpoint, row, deadline):
                save(output / "checkpoint.json", checkpoint)
        cfg.event_sink = sink
        result = solve_benders(data, scenarios, cfg)
    save(output / "result.json", result)


def reference_worker(directory, output, deadline):
    from .full_model import FullConfig, solve_full
    data, scenarios = load_case(directory, "n32-stoch")
    result = solve_full(data, scenarios, FullConfig(
        time_limit=max(.001, deadline-time.perf_counter()-3.), mip_gap=1e-6, output_flag=0))
    try:
        row = result_to_dict({k: v for k, v in vars(result).items() if k != "model"})
        row["certified"] = bool(result.model.Status == 2 and result.mip_gap <= 1e-6)
        if math.isfinite(result.objective):
            row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
        row["clock"] = time.perf_counter()
        save(output / "result.json", row)
    finally:
        result.model.dispose()


def worker(args):
    import gurobipy as gp
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 7)
    if args.worker == "reference":
        reference_worker(args.output_dir, args.output_dir / "references/n32-stoch", args.deadline)
    elif args.worker == "run":
        definition = next(r for r in read(args.output_dir / "manifest.json")["runs"] if r["id"] == args.identifier)
        if definition["hours"] is None:
            definition["hours"] = read(args.output_dir / "selection.json")["chosen_hours"]
        end_worker(args.output_dir, definition, args.output_dir / "runs" / definition["id"], args.deadline)
    elif args.worker == "report":
        report(args.output_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # Kaiwu housekeeping threads must not extend a finished trial.


def supervise(directory, output, kind, identifier, seconds, phase_deadline):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = min(phase_deadline, started+seconds)
    if deadline <= started:
        save(output / "state.json", dict(status="phase_budget_exhausted"))
        return
    save(output / "state.json", dict(status="running", started_clock=started, deadline=deadline))
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "suc_benders.gap_closure_experiment", "--output-dir", str(directory),
               "--worker", kind, "--identifier", identifier, "--deadline", str(deadline)]
    with (output / "console.txt").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        reason = "completed"
        while process.poll() is None:
            if time.perf_counter() >= deadline:
                reason = "time_limit"
                process.terminate()
                break
            time.sleep(.05)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if reason == "completed" and process.returncode:
            reason = "failed"
    save(output / "state.json", dict(status=reason, exitcode=process.returncode, started_clock=started,
                                      deadline=deadline, wall_seconds=time.perf_counter()-started))


def reference_record(directory, case):
    row = read(directory / "references" / case / "result.json")
    state = read(directory / "references" / case / "state.json")
    row["certified"] = bool(row.get("certified") and state.get("status") == "completed"
        and row.get("objective") is not None and row.get("lower_bound") is not None
        and row.get("mip_gap") is not None and row["mip_gap"] <= 1e-6
        and row["lower_bound"] <= row["objective"]+1e-5
        and (row.get("validation") or {}).get("feasible_solution")
        and ("clock" not in row or row["clock"] <= state.get("deadline", -math.inf)))
    return row


def statistics(directory, definition):
    folder = directory / "runs" / definition["id"]
    state = read(folder / "state.json")
    raw = events(folder / "events.jsonl")
    deadline = state.get("deadline", -math.inf)
    rows = [r for r in raw if r.get("clock", math.inf) <= deadline]
    evaluations = [r for r in rows if r["kind"] == "evaluation" and r.get("feasible")]
    data, scenarios = load_case(directory, definition["case"])
    errors, checked = [], []
    for row in evaluations:
        try:
            check = validate_schedule(data, scenarios, schedule_object(row))
            if not check["feasible_solution"]:
                raise AssertionError("no feasible solution")
            checked.append((row, check))
        except (AssertionError, KeyError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    best = min((r for r, _ in checked), key=lambda r:r["objective"], default=None)
    bounds = [r for r in rows if r["kind"] == "bound_update"
              and r.get("reliable_lower") is not None and math.isfinite(r["reliable_lower"])]
    bound = max(bounds, key=lambda r:r["reliable_lower"], default=None)
    lower = bound["reliable_lower"] if bound else None
    reference = reference_record(directory, definition["case"])
    ref = reference.get("objective") if reference["certified"] else None
    cost = best["objective"] if best else None
    uppers = [r["coordination_upper"] for r in bounds
              if r.get("coordination_upper") is not None and math.isfinite(r["coordination_upper"])]
    upper = min(uppers) if uppers else None
    for limit in (cost, ref, upper):
        if limit is not None and lower is not None and lower > limit+1e-5:
            errors.append("certified lower exceeds feasible/coordination upper")
    result = read(folder / "result.json")
    if result.get("diagnostics", {}).get("failure_stage"):
        errors.append(result["diagnostics"]["failure_stage"])
    samplings = [r for r in rows if r["kind"] == "sampling"]
    native = [r for r in rows if r["kind"] == "native_candidate"]
    inner = [r for r in rows if r["kind"] == "inner_iteration"]
    outer = [r for r in rows if r["kind"] == "iteration"]
    initial = min((r["objective"] for r, _ in checked if r["iteration"] == 0), default=None)
    valid = bool(bound and best and not errors and state.get("status") in ("completed", "time_limit"))
    gap = (ref-lower)/max(1., abs(ref)) if ref is not None and lower is not None and not errors else None
    hours = definition["hours"] or read(directory / "selection.json").get("chosen_hours")
    time_to = {}
    for threshold in (.01, .005):
        hits = [r["clock"]-state["started_clock"] for r in bounds
                if ref is not None and (ref-r["reliable_lower"])/max(1., abs(ref)) <= threshold]
        time_to[str(threshold)] = min(hits) if hits else None
    return dict(**{**definition, "hours": hours}, state=state, solver_status=result.get("status"),
        initial_cost=initial, final_cost=cost, best_source=best.get("source") if best else None,
        best_origins=best.get("candidate_origins", []) if best else [], reliable_lower=lower,
        reference_cost=ref, reference_certified=reference["certified"], paper_gap=gap,
        reference_cost_difference=(cost-ref)/max(1., abs(ref)) if cost is not None and ref is not None else None,
        primal_gap=(cost-lower)/max(1., abs(cost)) if cost is not None and lower is not None and not errors else None,
        coordination_upper=upper, numerical_audit_passed=valid, errors=errors,
        validated_schedules=len(checked), max_residual_mw=max((c["max_residual_mw"] for _, c in checked), default=None),
        completed_outer_rounds=len(outer), multiplier_updates=sum(not r.get("converged", False) for r in
            result.get("diagnostics", {}).get("outer", []) if r.get("dual_interval") is not None),
        attempted_blocks=len([r for r in rows if r["kind"] == "block_start"]),
        converged_blocks=sum(r.get("converged", False) for r in inner),
        native_calls=len([r for r in rows if r["kind"] == "native_start"]),
        native_accepted=sum(r["accepted"] for r in native), native_seconds=sum(r.get("seconds", 0.) for r in native),
        raw_samples=sum(r["raw_candidates"] for r in samplings),
        raw_discrete_feasible=sum(r["raw_discrete_feasible"] for r in samplings),
        time_to_gap=time_to, late_events=len(raw)-len(rows), result_saved=bool(result),
        allocated_seconds=deadline-state["started_clock"] if "started_clock" in state else None,
        bound_source=bound.get("bound_source") if bound else None,
        contribution={source: sum(max(0., r["incumbent_before"]-r["objective"])
                     if r.get("incumbent_before") is not None else 0. for r, _ in checked
                     if r.get("improved") and r["source"] == source)
                     for source in {r["source"] for r, _ in checked}})


def select_hours(runs, reference_cost):
    eligible, excluded = [], []
    for run in runs:
        if not run["numerical_audit_passed"]:
            excluded.append(dict(hours=run["hours"], reason="numerical_audit"))
        elif run["coordination_upper"] is not None and run["coordination_upper"] < .99*reference_cost-1e-5:
            excluded.append(dict(hours=run["hours"], reason="certified_dual_ceiling"))
        else:
            eligible.append(run)
    if not eligible:
        return dict(chosen_hours=None, reason="no_eligible_block_length", excluded=excluded)
    maximum = max(r["reliable_lower"] for r in eligible)
    chosen = min(r["hours"] for r in eligible if maximum-r["reliable_lower"] <= 1e-5)
    return dict(chosen_hours=chosen, rule="largest_on_time_certified_lower_then_shortest_hours", excluded=excluded)


def freeze(directory, historical, prior=None):
    directory.mkdir(parents=True)
    for case in ("n3-det", "n10-det", "n32-det", "n32-stoch"):
        save(directory / "inputs" / case / "units.json", case_input(int(case.split('-')[0][1:])))
        save(directory / "inputs" / case / "scenarios.json", read(ROOT / "data/rts24" /
             ("stochastic.json" if case.endswith("stoch") else "deterministic.json")))
    for case in ("n3-det", "n10-det", "n32-det"):
        for filename in ("units.json", "scenarios.json"):
            original = historical / "inputs" / case / filename
            if read(original) != read(directory / "inputs" / case / filename):
                raise ValueError(f"historical input mismatch: {original}")
            target = directory / "historical" / case / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        shutil.copyfile(historical / "references" / case / "result.json", directory / "historical" / case / "reference.json")
    source = list((ROOT / "suc_benders").glob("*.py"))+list((ROOT / "tests").glob("*.py"))
    source += list((ROOT / "data/rts24").glob("*.json"))+[ROOT / "requirements.txt"]
    for path in source:
        target = directory / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = dict(protocol="gap-closure-v2-deferred-recovery", runs=definitions(), seeds=SEEDS, hours=HOURS,
        total_seconds=TOTAL_SECONDS, phase_seconds=PHASE_SECONDS, primary_gap=.01, challenge_gap=.005,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty_paths=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        python=sys.version, dependencies={k: importlib.metadata.version(k) for k in ("numpy", "gurobipy", "kaiwu", "matplotlib")},
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source},
        input_sha256={p.relative_to(directory).as_posix(): sha(p) for p in (directory / "inputs").rglob("*.json")},
        historical_sha256={p.relative_to(directory).as_posix(): sha(p) for p in (directory / "historical").rglob("*.json")},
        configs={r["id"]: asdict(configuration(r)) for r in definitions()},
        selected_hours_pending=True)
    if prior is not None:
        previous = read(prior / "manifest.json")
        prior_state = read(prior / "state.json")
        prior_seconds = prior_state["seconds"]
        if not 0 <= prior_seconds < TOTAL_SECONDS:
            raise ValueError("prior attempt exhausted the total budget")
        for filename in ("units.json", "scenarios.json"):
            if read(prior / "inputs/n32-stoch" / filename) != read(directory / "inputs/n32-stoch" / filename):
                raise ValueError("cached stochastic reference input mismatch")
        if previous["source_sha256"]["suc_benders/full_model.py"] != sha(ROOT / "suc_benders/full_model.py"):
            raise ValueError("cached stochastic reference model mismatch")
        if not reference_record(prior, "n32-stoch")["certified"]:
            raise ValueError("prior stochastic reference is not certified")
        for filename in ("result.json", "state.json"):
            target = directory / "references/n32-stoch" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prior / "references/n32-stoch" / filename, target)
        row = read(directory / "references/n32-stoch/result.json")
        data, scenarios = load_case(directory, "n32-stoch")
        row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
        save(directory / "references/n32-stoch/result.json", row)
        manifest.update(prior_attempt=str(prior), prior_seconds=prior_seconds,
            prior_archive_sha256=sha(prior / "SHA256SUMS.txt"),
            cached_reference_sha256=sha(prior / "references/n32-stoch/result.json"),
            available_seconds=TOTAL_SECONDS-prior_seconds,
            amendment="User authorized deferred full-day evaluation; original attempts retained and budget debited.")
    else:
        manifest.update(prior_seconds=0., available_seconds=TOTAL_SECONDS)
    return manifest


def validate_historical(directory):
    for case in ("n3-det", "n10-det", "n32-det"):
        row = read(directory / "historical" / case / "reference.json")
        data, scenarios = load_case(directory, case)
        row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
        row["certified"] = bool(row.get("certified") and row.get("mip_gap", math.inf) <= 1e-6)
        save(directory / "references" / case / "result.json", row)
        save(directory / "references" / case / "state.json", dict(status="completed", historical=True))
        if not reference_record(directory, case)["certified"]:
            raise ValueError(f"uncertified historical reference: {case}")


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in columns} for r in rows)


def plot(directory, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, filename, label in (("paper_gap", "paper-gap.png", "Paper-style gap (%)"),
                                    ("reliable_lower", "certified-lower.png", "Certified lower bound")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for definition in read(directory / "manifest.json")["runs"]:
            if definition["stage"] != "confirmation":
                continue
            values = [r for r in trajectories if r["run"] == definition["id"] and r[metric] is not None]
            if values:
                ax.step([r["seconds"] for r in values], [r[metric]*(100 if metric == "paper_gap" else 1) for r in values],
                        where="post", label=f"seed {definition['seed']}")
        if metric == "paper_gap":
            ax.axhline(1., color="grey", linestyle="--", label="1% target")
            ax.axhline(.5, color="grey", linestyle=":", label="0.5% challenge")
        ax.set(xlabel="Wall seconds from process launch", ylabel=label)
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def report(directory):
    manifest = read(directory / "manifest.json")
    hash_errors = []
    for field, prefix in (("source_sha256", "source"), ("input_sha256", ""), ("historical_sha256", "")):
        for path, digest in manifest[field].items():
            if sha(directory / prefix / path) != digest:
                hash_errors.append(path)
    rows = [statistics(directory, definition) for definition in manifest["runs"]]
    trajectories, blocks = [], []
    for row in rows:
        state = row["state"]
        raw = events(directory / "runs" / row["id"] / "events.jsonl")
        for event in raw:
            if event.get("clock", math.inf) > state.get("deadline", -math.inf):
                continue
            if event["kind"] == "bound_update":
                lower = event["reliable_lower"]
                ref = row["reference_cost"]
                trajectories.append(dict(run=row["id"], stage=row["stage"], method=row["method"], seed=row["seed"],
                    hours=row["hours"], seconds=event["clock"]-state["started_clock"],
                    outer=event["outer"], block=event["block"], inner=event["inner"], reliable_lower=lower,
                    paper_gap=(ref-lower)/max(1., abs(ref)) if ref is not None and not row["errors"] else None,
                    bound_source=event["bound_source"]))
            elif event["kind"] == "inner_iteration":
                blocks.append(dict(run=row["id"], **event))
    confirmations = [r for r in rows if r["stage"] == "confirmation"]
    primary = len(confirmations) == 3 and not hash_errors and all(
        r["numerical_audit_passed"] and r["reference_certified"] and r["paper_gap"] is not None
        and r["paper_gap"] <= .01 and r["allocated_seconds"] is not None
        and r["allocated_seconds"] >= 600.-1e-5 for r in confirmations)
    challenge = primary and all(r["paper_gap"] <= .005 for r in confirmations)
    controller = read(directory / "state.json")
    primary = bool(primary and controller.get("within_total_budget", False))
    challenge = bool(challenge and primary)
    summary = dict(protocol=manifest["protocol"], execution_commit=manifest["commit"],
        source_hashes_valid=not hash_errors, hash_errors=hash_errors, selection=read(directory / "selection.json"),
        reference={c: reference_record(directory, c) for c in ("n3-det", "n10-det", "n32-det", "n32-stoch")},
        runs=rows, controller=controller, primary_passed=primary, challenge_passed=challenge)
    save(directory / "summary.json", summary)
    save(directory / "audit.json", dict(passed=not hash_errors and not any(r["errors"] for r in rows),
        primary_passed=primary, challenge_passed=challenge, hash_errors=hash_errors,
        all_slots_accounted_for=all(r["state"].get("status") in
            ("completed", "time_limit", "failed", "skipped", "phase_budget_exhausted") for r in rows),
        confirmed_valid_schedules=[r["validated_schedules"] for r in confirmations],
        within_total_budget=controller.get("within_total_budget", False)))
    write_csv(directory / "runs.csv", rows, ["id", "stage", "case", "hours", "method", "seed", "solver_status",
        "final_cost", "reliable_lower", "reference_cost", "paper_gap", "reference_cost_difference", "primal_gap",
        "coordination_upper", "numerical_audit_passed", "completed_outer_rounds", "multiplier_updates",
        "native_calls", "native_accepted", "native_seconds", "raw_samples", "raw_discrete_feasible", "best_source"])
    write_csv(directory / "blocks.csv", blocks, ["run", "clock", "outer", "block", "inner", "lower", "upper", "cuts", "converged"])
    write_csv(directory / "curves.csv", trajectories, ["run", "stage", "method", "seed", "hours", "seconds", "outer",
                                                      "block", "inner", "reliable_lower", "paper_gap", "bound_source"])
    plot(directory, trajectories)
    def pct(value):
        return "NA" if value is None else f"{100*value:.4f}%"
    lines = ["# RTS-24 随机 SUC Gap 补足实验", "",
        f"主目标（三次 600 秒均 ≤1%）：{primary}；挑战目标（均 ≤0.5%）：{challenge}。",
        f"选择分块：{summary['selection'].get('chosen_hours', 'NA')}h；控制状态：{controller.get('status', 'NA')}。", "",
        "论文口径 Gap=(独立参考可行成本−认证下界)/参考成本。参考方案和成本没有进入搜索；截止后事件不计分。",
        "全部运行串行、单线程；正式 40 次外层／200 次内层，容差 1e-4；SA 参数及数据未调优。", "",
        "| 阶段 | 方法/种子 | h | 状态 | 成本 | 认证下界 | 论文gap | 成本高于参考 | 原问题gap | 外层完成 | 经典调用/接受 | 来源 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['stage']} | {row['method']}/{row['seed']} | {row['hours']} | {row['solver_status'] or row['state'].get('status')} | "
            f"{row['final_cost']} | {row['reliable_lower']} | {pct(row['paper_gap'])} | {pct(row['reference_cost_difference'])} | "
            f"{pct(row['primal_gap'])} | {row['completed_outer_rounds']} | {row['native_calls']}/{row['native_accepted']} | {row['best_source']} |")
    lines += ["", "经典点经原局部调度及全日恢复验证；来源 native_master／mixed_native 单列。",
        "原始样本可行数、经典耗时、贡献金额、截止后事件及失败原因见 summary.json；每个块和每组乘子分别记录。",
        "达到论文口径目标与 DUAL_CONVERGED 分开判断；超时不等于理论对偶上限。", "",
        "![论文口径 Gap](paper-gap.png)", "", "![认证下界](certified-lower.png)", "",
        "源码、输入、历史参考和逐文件 SHA256 随实验冻结；--report-only 可重建报告。"]
    (directory / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run(directory, historical, prior=None):
    if directory.exists():
        raise ValueError("use a new output directory")
    started = time.perf_counter()
    manifest = freeze(directory, historical, prior)
    deadline = started+manifest["available_seconds"]
    manifest.update(started_clock=started, started_unix=time.time(), deadline=deadline)
    save(directory / "manifest.json", manifest)
    save(directory / "state.json", dict(status="running", within_total_budget=True))
    stop_reason = None
    try:
        validate_historical(directory)
        if prior is None:
            supervise(directory, directory / "references/n32-stoch", "reference", "n32-stoch", 600,
                      min(deadline-180, started+PHASE_SECONDS["reference"]))
        if not reference_record(directory, "n32-stoch")["certified"]:
            stop_reason = "reference_not_certified"
        cumulative_budget = 0
        for stage, budget in PHASE_SECONDS.items():
            cumulative_budget += budget
            if stage in ("reference", "audit") or stop_reason:
                continue
            phase_deadline = min(deadline-180, started+cumulative_budget-manifest["prior_seconds"])
            for definition in [r for r in manifest["runs"] if r["stage"] == stage]:
                save(directory / "progress.json", dict(stage=stage, run=definition["id"], started_clock=time.perf_counter()))
                supervise(directory, directory / "runs" / definition["id"], "run", definition["id"],
                          definition["seconds"], phase_deadline)
                row = statistics(directory, definition)
                print(f"finished {definition['id']}: paper_gap={row['paper_gap']} lower={row['reliable_lower']}", flush=True)
            if stage == "screening":
                screened = [statistics(directory, r) for r in manifest["runs"] if r["stage"] == stage]
                selection = select_hours(screened, reference_record(directory, "n32-stoch")["objective"])
                save(directory / "selection.json", selection)
                if selection["chosen_hours"] is None:
                    stop_reason = selection["reason"]
            elif stage == "sanity":
                sanity = [statistics(directory, r) for r in manifest["runs"] if r["stage"] == stage]
                if not all(r["numerical_audit_passed"] and r["reference_cost"] is not None
                           and abs(r["final_cost"]-r["reference_cost"]) <= 1e-4 for r in sanity):
                    stop_reason = "sanity_failed"
    except Exception as exc:
        stop_reason = f"controller_error: {type(exc).__name__}: {exc}"
    for definition in manifest["runs"]:
        state_path = directory / "runs" / definition["id"] / "state.json"
        if not state_path.exists():
            save(state_path, dict(status="skipped", reason=stop_reason or "budget"))
    save(directory / "state.json", dict(status="search_finished", stop_reason=stop_reason,
        seconds=time.perf_counter()-started, total_attempt_seconds=manifest["prior_seconds"]+time.perf_counter()-started,
        within_total_budget=time.perf_counter() <= deadline))
    supervise(directory, directory / "report-worker", "report", "all", 180, deadline)
    report_state = read(directory / "report-worker/state.json")
    state = dict(status="completed" if report_state.get("status") == "completed" else "report_incomplete",
        stop_reason=stop_reason, seconds=time.perf_counter()-started,
        total_attempt_seconds=manifest["prior_seconds"]+time.perf_counter()-started,
        within_total_budget=time.perf_counter() <= deadline)
    save(directory / "state.json", state)
    # Update the controller envelope without rerunning expensive physical checks.
    summary, audit = read(directory / "summary.json"), read(directory / "audit.json")
    if summary:
        summary["controller"] = state
        summary["primary_passed"] &= state["within_total_budget"] and state["status"] == "completed"
        summary["challenge_passed"] &= summary["primary_passed"]
        save(directory / "summary.json", summary)
        audit.update(primary_passed=summary["primary_passed"], challenge_passed=summary["challenge_passed"],
                     within_total_budget=state["within_total_budget"])
        save(directory / "audit.json", audit)
        readme = directory / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace("控制状态：search_finished", f"控制状态：{state['status']}"),
                          encoding="utf-8")
    checksum(directory)
    print(f"completed {directory}: {state}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, help="raw SA confirmation archive containing inputs/ and references/")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prior-attempt", type=Path, help="retained attempt to debit; reuse its independently certified identical stochastic reference")
    parser.add_argument("--worker", choices=("reference", "run", "report"), help=argparse.SUPPRESS)
    parser.add_argument("--identifier", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
    elif args.report_only:
        report(args.output_dir)
        checksum(args.output_dir)
    else:
        if args.historical_dir is None:
            parser.error("--historical-dir is required for frozen deterministic references")
        run(args.output_dir, args.historical_dir.resolve(), args.prior_attempt.resolve() if args.prior_attempt else None)


if __name__ == "__main__":
    main()
