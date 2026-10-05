"""Prespecified one-hour, serial SA scale diagnostic. No production defaults change."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .block_master_experiment import audit_encoding, native_checks, snapshot_spec
from .block_master_model import RecordedQubo, continuous_relaxation, point_record, priced_blocks, solve_master
from .candidates import repair_commitment
from .data import load_suc_data
from .master import QuantumConfig
from .paper_model import Cut, IndexedUC, key
from .solver import BendersConfig, _construct_feasible_candidate, _fixed_cost, result_to_dict
from .subagent import publish, sha
from .subproblem import solve_scenario
from .validation import schedule_object, validate_schedule


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 100007, 200007)
SUBSETS = {3: (3, 9, 20), 5: (1, 3, 9, 12, 20),
           10: (1, 3, 9, 12, 15, 20, 22, 24, 30, 32), 32: tuple(range(1, 33))}
CASES = tuple(f"n{n}-det" for n in SUBSETS) + ("n32-stoch",)
END_TO_END = ((3, 1, "det", "varying"), (10, 1, "det", "varying"),
              (32, 1, "det", "varying"), (32, 3, "det", "varying"),
              (32, 3, "det", "fixed"), (32, 3, "stoch", "varying"))
PHASE_SECONDS = {"A": 480, "B": 1080, "C": 1620, "D": 420}


def read(path):
    for attempt in range(6):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        except PermissionError:
            # Windows can briefly deny a reader during an atomic file replacement.
            if attempt == 5:
                raise
            time.sleep(.01)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish(path, result_to_dict(value))


def append(stream, row):
    stream.write(json.dumps(result_to_dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def events(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return rows


def config(seed=7, mode="varying", seconds=None, hours=3):
    return BendersConfig(max_inner_iterations=60, block_hours=hours, time_limit_seconds=seconds,
                         sa_seed_per_call=mode == "varying", quantum=QuantumConfig(kaiwu_seed=seed))


def seed_for(seed, mode, call):
    return seed + call if mode == "varying" else seed


def case_input(n):
    original = read(ROOT / "data/rts24/units.json")
    ids = SUBSETS[n]
    factor = sum(original["pmax"][i-1] for i in ids)/sum(original["pmax"])
    result = {k: ([v[i-1] for i in ids] if isinstance(v, list) and k != "base_load" else v)
              for k, v in original.items()}
    result.update(n_units=n, base_load=[v*factor for v in original["base_load"]])
    return result


def definitions():
    snapshots = [dict(id=f"n{n}-h{h}-{role}", case=f"n{n}-det", n=n, hours=h,
                      block=0 if role == "first" else 11//h, role=role)
                 for n in SUBSETS for h in (1, 2, 3) for role in ("first", "peak")]
    trials = [dict(id=f"{s['id']}/{seed}-{mode}", snapshot=s["id"], seed=seed, mode=mode)
              for s in snapshots for seed in SEEDS for mode in ("fixed", "varying")]
    runs = [dict(id=f"n{n}-{profile}-h{h}-{mode}-{seed}", case=f"n{n}-{profile}",
                 hours=h, seed=seed, mode=mode)
            for n, h, profile, mode in END_TO_END for seed in SEEDS]
    return dict(snapshots=snapshots, trials=trials, runs=runs)


def load_case(directory, case):
    return load_suc_data(directory / "inputs" / case / "units.json",
                         directory / "inputs" / case / "scenarios.json")


def evaluate(data, scenarios, candidate, deadline):
    if candidate is None:
        return None
    solutions = []
    for scenario in scenarios:
        if time.perf_counter() >= deadline:
            return None
        r = solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w, deadline=deadline, threads=1)
        if not r.feasible:
            return None
        solutions.append(dict(name=scenario.name, objective=r.objective, p=r.p, r=r.r))
    objective = _fixed_cost(data, candidate.u, candidate.v, candidate.w) + sum(
        s.probability*r["objective"] for s, r in zip(scenarios, solutions))
    row = result_to_dict(dict(objective=objective, u=candidate.u, v=candidate.v, w=candidate.w,
                             source=candidate.source, scenarios=solutions))
    row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
    return row


def initialize(directory, case, deadline):
    data, scenarios = load_case(directory, case)
    uc = IndexedUC(data, scenarios)
    relaxation = continuous_relaxation(uc, deadline)
    u = np.array([[round(relaxation["point"][key("u", i, t)]) for t in range(data.horizon)]
                  for i in range(data.n_units)])
    rounded = repair_commitment(data, scenarios, u)
    if rounded is not None:
        rounded.source = "relaxation_rounding"
    checked = [evaluate(data, scenarios, c, deadline) for c in
               (rounded, _construct_feasible_candidate(data, scenarios))]
    checked = [c for c in checked if c is not None]
    if not checked:
        raise ValueError("initialization has no validated schedule")
    initial = min(checked, key=lambda c: c["objective"])
    save(directory / "cases" / case / "initial.json", dict(best=initial, candidates=checked,
         relaxation={**relaxation, "point": point_record(relaxation["point"])}))
    for definition in definitions()["snapshots"]:
        if definition["case"] != case:
            continue
        if time.perf_counter() >= deadline:
            return
        folder = directory / "snapshots" / definition["id"]
        block = priced_blocks(uc, definition["hours"], relaxation["multipliers"])[0][definition["block"]]
        point = {k: int(initial[k[0]][k[1]][k[2]]) for k in block.bkeys}
        dispatch = block.dispatch(point, block.costs, deadline)
        if dispatch is None or not dispatch["feasible"]:
            save(folder / "failure.json", dict(status="initial_block_dispatch_failed"))
            continue
        cuts = [dispatch["cut"]]
        analytic_lo, analytic_hi = block.objective_bounds(block.costs)
        lp = block.relaxation(block.costs, cuts, deadline)
        if lp is None:
            raise ValueError("initial master relaxation unavailable")
        lower, upper = max(analytic_lo, lp-1e-5), min(dispatch["cost"], analytic_hi)
        cfg = config()
        master = RecordedQubo(block, block.costs, cuts, lower, upper, cfg)
        witness = audit_encoding(master, block, dict(upper_point=point, cuts=cuts))
        record = dict(**definition, multipliers=relaxation["multipliers"], lower=lower, upper=upper,
                      cuts=[dict(a=point_record(c.a), c=c.c, feasible=c.feasible) for c in cuts],
                      spec=snapshot_spec(block, cuts, lower, upper, cfg), witness=witness)
        save(folder / "snapshot.json", record)
        np.savez_compressed(folder / "qubo.npz", matrix=master.matrix,
                            offset=master.model.get_offset(), names=np.asarray(master.names))
        nz = np.abs(master.matrix[master.matrix != 0])
        # Sampling metadata intentionally excludes witnesses, multipliers and reference solutions.
        save(folder / "sampling.json", dict(energy_unit=master.energy_unit,
             quantum=asdict(cfg.quantum), bits=len(master.names), commitment_bits=len(master.vars),
             objective_bits=len(master.zvars), auxiliary_bits=len(master.names)-len(master.vars)-len(master.zvars),
             fixed_bits=len(master.fixed), coefficient_min=float(nz.min()) if nz.size else None,
             coefficient_max=float(nz.max()) if nz.size else None,
             normalized_energy_unit=master.energy_unit/max(1., float(nz.max()) if nz.size else 1.)))
        save(folder / "frozen.json", {p: sha(folder / p) for p in ("snapshot.json", "qubo.npz", "sampling.json")})


def frozen_sampler_input(folder):
    # Only these two files cross into the SA process. No native oracle or witness channel.
    metadata = read(folder / "sampling.json")
    with np.load(folder / "qubo.npz", allow_pickle=False) as payload:
        matrix = payload["matrix"].copy()
    return matrix, metadata


def restore_snapshot(directory, identifier):
    folder = directory / "snapshots" / identifier
    frozen = read(folder / "frozen.json")
    if not frozen or any(sha(folder / p) != digest for p, digest in frozen.items()):
        raise ValueError("snapshot hash mismatch or missing freeze")
    row = read(folder / "snapshot.json")
    data, scenarios = load_case(directory, row["case"])
    block = priced_blocks(IndexedUC(data, scenarios), row["hours"], row["multipliers"])[0][row["block"]]
    cuts = [Cut({tuple(k): v for k, v in c["a"]}, c["c"], c["feasible"]) for c in row["cuts"]]
    master = RecordedQubo(block, block.costs, cuts, row["lower"], row["upper"], config())
    with np.load(folder / "qubo.npz", allow_pickle=False) as payload:
        if (not np.array_equal(master.matrix, payload["matrix"]) or master.names != payload["names"].tolist()
                or master.model.get_offset() != float(payload["offset"])):
            raise ValueError("reconstructed snapshot differs from frozen matrix")
    return block, cuts, master


def sample_worker(directory, definition, output, deadline):
    from .common import get_kaiwu
    from .quantum import solve_qubo
    get_kaiwu()  # Import/license startup is charged to the phase, not to SA time.
    matrix, meta = frozen_sampler_input(directory / "snapshots" / definition["snapshot"])
    spent = 0.
    with (output / "batches.jsonl").open("w", encoding="utf-8") as stream:
        for call in range(20):
            remaining = min(5.-spent, deadline-time.perf_counter())
            if remaining <= 0:
                break
            seed = seed_for(definition["seed"], definition["mode"], call)
            qconfig = replace(QuantumConfig(**meta["quantum"]), kaiwu_seed=seed)
            started = time.perf_counter()
            call_deadline = started+remaining
            save(output / "active.json", dict(deadline=call_deadline, phase="sampling", call=call, seed=seed,
                                             completed_sa_seconds=spent))
            bits = solve_qubo(matrix, qconfig, energy_unit=meta["energy_unit"])
            elapsed = time.perf_counter()-started
            if not completed_in_time(started+elapsed, call_deadline):
                break
            save(output / "active.json", dict(deadline=deadline, phase="serializing"))
            spent += elapsed
            append(stream, dict(call=call, seed=seed, seconds=elapsed, cumulative_sa_seconds=spent,
                                samples=[np.asarray(b, dtype=int).tolist() for b in bits]))
    save(output / "result.json", dict(completed_calls=len(events(output / "batches.jsonl")), sa_seconds=spent))


def completed_in_time(ended, deadline):
    return ended <= deadline


def oracle_worker(directory, identifier, output, deadline, kind):
    block, cuts, master = restore_snapshot(directory, identifier)
    if kind == "native":
        row = solve_master(block, cuts, min(time.perf_counter()+3., deadline-.3))
        point = row.pop("point")
        row["native_checks"] = native_checks(block, cuts, point, row["theta"]) if point is not None else None
        row["point"] = point_record(point)
        save(output / "result.json", row)
        return
    import gurobipy as gp
    model = gp.Model("frozen_qubo_diagnostic")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 7
    model.Params.NumericFocus = 3
    model.Params.MIPGap = 1e-6
    x = model.addMVar(len(master.names), vtype=gp.GRB.BINARY)
    model.setObjective(x @ master.matrix @ x + master.model.get_offset())
    model.Params.TimeLimit = max(.001, min(7., deadline-time.perf_counter()-.3))
    model.optimize()
    row = dict(status=int(model.Status), seconds=model.Runtime, reported_bound=float(model.ObjBound),
               numeric_caution="Wide coefficient ranges: OPTIMAL alone is not an exact arithmetic certificate.")
    if model.SolCount:
        bits = np.rint(x.X).astype(int)
        row.update(bits=bits, reported_energy=model.ObjVal,
                   recomputed_energy=float(bits @ master.matrix @ bits + master.model.get_offset()),
                   sample=check_sample(block, cuts, master, bits))
    save(output / "result.json", row)
    model.dispose()


def reference_worker(directory, case, output, deadline):
    from .full_model import FullConfig, solve_full
    data, scenarios = load_case(directory, case)
    result = solve_full(data, scenarios, FullConfig(time_limit=max(.001, deadline-time.perf_counter()-1.),
                                                  mip_gap=1e-6, output_flag=0))
    row = result_to_dict({k: v for k, v in vars(result).items() if k != "model"})
    if math.isfinite(result.objective):
        row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
    row["certified"] = result.model.Status == 2
    save(output / "result.json", row)
    result.model.dispose()


def end_worker(directory, definition, output, deadline):
    from .solver import solve_benders
    data, scenarios = load_case(directory, definition["case"])
    cfg = config(definition["seed"], definition["mode"], max(.001, deadline-time.perf_counter()-2.), definition["hours"])
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        cfg.event_sink = lambda row: append(stream, row)
        result = solve_benders(data, scenarios, cfg)
    save(output / "result.json", result)


def supervise(directory, output, kind, identifier, seconds, phase_deadline):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = min(phase_deadline, started+seconds)
    if deadline <= started:
        save(output / "state.json", dict(status="phase_budget_exhausted"))
        return
    save(output / "state.json", dict(status="running", deadline=deadline))
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    args = [sys.executable, "-m", "suc_benders.sa_scale_experiment", "--output-dir", str(directory),
            "--worker", kind, "--identifier", identifier, "--deadline", str(deadline)]
    with (output / "console.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        reason = "completed"
        try:
            while process.poll() is None:
                active = read(output / "active.json")
                if time.perf_counter() >= min(deadline, active.get("deadline", deadline)):
                    reason = "sampling_budget_exhausted" if active.get("phase") == "sampling" else "time_limit"
                    break
                time.sleep(.02)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
    if reason == "completed" and process.returncode:
        reason = "failed"
    save(output / "state.json", dict(status=reason, exitcode=process.returncode,
                                     wall_seconds=time.perf_counter()-started, deadline=deadline))


def worker(args):
    import gurobipy as gp
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 7)
    directory, identifier = args.output_dir, args.identifier
    defs = definitions()
    if args.worker == "init":
        initialize(directory, identifier, args.deadline)
    elif args.worker == "reference":
        reference_worker(directory, identifier, directory / "references" / identifier, args.deadline)
    elif args.worker in ("native", "qubo"):
        oracle_worker(directory, identifier, directory / "snapshots" / identifier / args.worker,
                      args.deadline, args.worker)
    elif args.worker == "sample":
        definition = next(t for t in defs["trials"] if t["id"] == identifier)
        sample_worker(directory, definition, directory / "trials" / identifier, args.deadline)
    elif args.worker == "end":
        definition = next(t for t in defs["runs"] if t["id"] == identifier)
        end_worker(directory, definition, directory / "end_to_end" / identifier, args.deadline)
    elif args.worker == "report":
        report(directory)
    # Kaiwu may leave SDK housekeeping threads alive after all results are flushed.
    # Isolated workers have no further work; do not charge library teardown to trials.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def check_sample(block, cuts, master, bits):
    point, theta = master.decode(bits)
    native = native_checks(block, cuts, point, theta)
    encoded = master.encoded_checks(bits)
    objective = float(theta + sum(block.costs[k]*point[k] for k in block.bkeys) + block.constant)
    return dict(raw_uvw=[int(point[k]) for k in block.bkeys], theta=float(theta),
                native=native, encoded=encoded,
                raw_objective=objective if native["feasible"] else None,
                energy=float(np.asarray(bits) @ master.matrix @ np.asarray(bits) + master.model.get_offset()))


def trial_statistics(samples, batches, reference):
    count = len(samples)
    best = min((s["raw_objective"] for s in samples if s["raw_objective"] is not None), default=None)
    certified = bool(reference.get("certified") and (reference.get("native_checks") or {}).get("feasible"))
    absolute = best-reference["objective"] if best is not None and certified else None
    relative = absolute/max(1., abs(reference["objective"])) if absolute is not None else None
    hashes = [hashlib.sha256(bytes(v for bits in b["samples"] for v in bits)).hexdigest() for b in batches]
    return dict(samples=count, completed_calls=len(batches), sa_seconds=sum(b["seconds"] for b in batches),
        unique_uvw=len({tuple(s["raw_uvw"]) for s in samples}),
        repeated_batch_fraction=(len(hashes)-len(set(hashes)))/len(hashes) if hashes else None,
        discrete_feasible_rate=sum(s["native"]["discrete"]["feasible"] for s in samples)/count if count else None,
        native_feasible_rate=sum(s["native"]["feasible"] for s in samples)/count if count else None,
        encoded_feasible_rate=sum(s["encoded"]["feasible"] for s in samples)/count if count else None,
        best_raw_objective=best, reference_certified=certified, absolute_gap=absolute, relative_gap=relative)


def end_statistics(directory, definition):
    folder = directory / "end_to_end" / definition["id"]
    result, rows = read(folder / "result.json"), events(folder / "events.jsonl")
    data, scenarios = load_case(directory, definition["case"])
    evaluations = [r for r in rows if r["kind"] == "evaluation" and r["feasible"]]
    checks = [validate_schedule(data, scenarios, schedule_object(r)) for r in evaluations]
    initial = min((r["objective"] for r in evaluations if r["iteration"] == 0), default=None)
    best = min(evaluations, key=lambda r: r["objective"]) if evaluations else None
    if result.get("objective") is not None:
        checks.append(validate_schedule(data, scenarios, schedule_object(result)))
    diagnostics = result.get("diagnostics", {})
    outer = diagnostics.get("outer", [])
    inner = [r for r in rows if r["kind"] == "inner_iteration"]
    starts = {(r["outer"], r["block"]) for r in rows if r["kind"] == "sa_start"}
    finished = {(r["outer"], r["block"]) for r in inner if r["converged"]}
    entered = max((b[0] for b in starts), default=0)
    lower = diagnostics.get("reliable_lower")
    if lower is None:
        lower = next((r["reliable_lower"] for r in reversed(rows) if r["kind"] == "iteration"), None)
    reference = read(directory / "references" / definition["case"] / "result.json")
    ref = reference.get("objective")
    cost = best["objective"] if best else None
    proposals = [r for r in rows if r["kind"] == "proposal"]
    sampling = [r for r in rows if r["kind"] == "sampling"]
    return dict(**definition, state=read(folder / "state.json"), solver_status=result.get("status"),
        initial_cost=initial, final_cost=cost, best_source=best["source"] if best else None,
        sa_with_recovery_improvement=initial-cost if initial is not None and cost is not None else None,
        reliable_lower=lower, relative_reference_cost_difference=(cost-ref)/abs(ref) if cost is not None and ref else None,
        primal_gap=(cost-lower)/max(1., abs(cost)) if cost is not None and lower is not None else None,
        paper_style_gap=(ref-lower)/max(1., abs(ref)) if ref is not None and lower is not None else None,
        raw_samples=sum(s["raw_candidates"] for s in sampling),
        raw_discrete_feasible=sum(s["raw_discrete_feasible"] for s in sampling),
        recovered_proposals=sum(bool(p["repaired_discrete_feasible"]) for p in proposals),
        attempted_blocks=len(starts), converged_blocks=len(finished),
        attempted_block_convergence_fraction=len(finished)/len(starts) if starts else None,
        first_outer_blocks_total=math.ceil(24/definition["hours"]),
        first_outer_blocks_converged=sum(o == 1 for o, b in finished),
        multiplier_updates=max(0, max(len(outer), entered)-1),
        inner_iterations=len(inner), validated_schedules=len(checks),
        max_residual_mw=max((c.get("max_residual_mw", 0.) for c in checks), default=None),
        timings=diagnostics.get("timings"), partial=result == {})


def report(directory):
    from collections import Counter
    manifest = read(directory / "manifest.json")
    for path, digest in manifest["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"source changed during experiment: {path}")
    for path, digest in manifest["input_sha256"].items():
        if sha(directory / path) != digest:
            raise ValueError(f"input changed during experiment: {path}")
    references = []
    for case in CASES:
        reference = read(directory / "references" / case / "result.json")
        if reference.get("objective") is not None:
            data, scenarios = load_case(directory, case)
            validate_schedule(data, scenarios, schedule_object(reference))
        references.append(dict(case=case, state=read(directory / "references" / case / "state.json"),
            objective=reference.get("objective"), lower_bound=reference.get("lower_bound"),
            mip_gap=reference.get("mip_gap"), certified=reference.get("certified", False)))
    trials, errors = [], []
    for definition in manifest["snapshots"]:
        folder = directory / "snapshots" / definition["id"]
        snapshot_trials = [t for t in manifest["trials"] if t["snapshot"] == definition["id"]]
        try:
            block, cuts, master = restore_snapshot(directory, definition["id"])
        except Exception as exc:
            errors.append(dict(snapshot=definition["id"], error=str(exc)))
            for t in snapshot_trials:
                trials.append(dict(**t, state={"status": "snapshot_unavailable"}))
            continue
        reference = read(folder / "native/result.json")
        for t in snapshot_trials:
            trial_folder = directory / "trials" / t["id"]
            batches = events(trial_folder / "batches.jsonl")
            samples, violations = [], Counter()
            for batch in batches:
                if batch["seed"] != seed_for(t["seed"], t["mode"], batch["call"]):
                    raise ValueError("unexpected sampling seed")
                for index, bits in enumerate(batch["samples"]):
                    s = check_sample(block, cuts, master, bits)
                    samples.append(dict(call=batch["call"], sample=index, **s))
                    violations.update(s["native"]["discrete"]["violations"].keys())
                    if s["native"]["discrete"]["capacity_violation_mw"] > 1e-6:
                        violations["capacity"] += 1
            stats = trial_statistics(samples, batches, reference)
            state = read(trial_folder / "state.json") or {"status": "not_run"}
            if stats["sa_seconds"] > 5.+1e-8 or len(batches) > 20:
                raise ValueError("sampling budget exceeded")
            save(trial_folder / "samples.json", samples)
            row = dict(**t, **stats, state=state, violations=dict(violations))
            trials.append(row)
            save(trial_folder / "summary.json", row)
    runs = [end_statistics(directory, d) for d in manifest["runs"]]
    groups = []
    for s in manifest["snapshots"]:
        for mode in ("fixed", "varying"):
            selected = [t for t in trials if t["snapshot"] == s["id"] and t["mode"] == mode]
            known = all(t.get("reference_certified") and t.get("samples", 0) > 0 for t in selected)
            gaps = [t["relative_gap"] for t in selected if t.get("relative_gap") is not None]
            passed = known and len(gaps) == 3 and all(-1e-6 <= g <= .01 for g in gaps)
            groups.append(dict(snapshot=s["id"], mode=mode, verdict="passed" if passed else "not_passed" if known else "inconclusive",
                feasible_seeds=sum(t.get("best_raw_objective") is not None for t in selected),
                median_gap=float(np.median(gaps)) if gaps else None, worst_gap=max(gaps) if gaps else None,
                median_unique=float(np.median([t.get("unique_uvw", 0) for t in selected])),
                median_native_rate=float(np.median([t["native_feasible_rate"] for t in selected
                                                  if t.get("native_feasible_rate") is not None]))
                if any(t.get("native_feasible_rate") is not None for t in selected) else None))
    summary = dict(protocol_complete=len(trials) == 144 and len(runs) == 18, snapshot_errors=errors,
                   references=references,
                   groups=groups, trials=trials, end_to_end=runs,
                   warning="Three-seed screening; failures and censored runs retained. Subset composition changes with scale.")
    save(directory / "summary.json", summary)
    lines = ["# 纯 SA 分级诊断：固定协议初筛", "", f"源码提交：`{manifest['commit']}`。",
             "3 个种子；24 小时；固定参数。初筛不代表统计显著性。完整原始结果见 summary.json。", "",
             "原始主问题可行率不包含辅助变量重优化或修复。NA 表示没有可行样本或参考未认证。", "",
             "| 快照 | 种子策略 | 有可行解种子 | 主问题可行率中位数 | 误差中位数 | 最差误差 | 初筛 |",
             "|---|---|---:|---:|---:|---:|---|"]
    def pct(v):
        return "NA" if v is None else f"{100*v:.3f}%"
    for g in groups:
        lines.append(f"| {g['snapshot']} | {g['mode']} | {g['feasible_seeds']}/3 | {pct(g['median_native_rate'])} | {pct(g['median_gap'])} | {pct(g['worst_gap'])} | {g['verdict']} |")
    lines += ["", "| 完整运行 | 状态 | SA及恢复成本改进 | 成本高于参考 | 原始上下界gap | 论文口径gap | 首轮完成块 | 乘子更新 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        improvement = "NA" if r["sa_with_recovery_improvement"] is None else f"{r['sa_with_recovery_improvement']:.3f}"
        lines.append(f"| {r['id']} | {r['solver_status'] or r['state'].get('status', 'not_run')} | {improvement} | {pct(r['relative_reference_cost_difference'])} | {pct(r['primal_gap'])} | {pct(r['paper_style_gap'])} | {r['first_outer_blocks_converged']}/{r['first_outer_blocks_total']} | {r['multiplier_updates']} |")
    lines += ["", "原始离散可行、原始主问题可行、编码约束可行分别统计。QUBO经典参考具有数值限制，不能仅凭OPTIMAL宣称精确全局证书。",
              "完整算法保留原有全日恢复；成本改进归为SA及恢复流程，初始化成本单列。DUAL_CONVERGED不表示原始问题最优。"]
    (directory / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run(directory, prior_attempt=None):
    if directory.exists():
        raise ValueError("output directory already exists; use a new directory")
    directory.mkdir(parents=True)
    started = time.perf_counter()
    prior_seconds = 0.
    if prior_attempt is not None:
        previous = read(prior_attempt / "state.json")
        if previous.get("status") != "infrastructure_aborted":
            raise ValueError("only an explicitly archived infrastructure abort can debit this run")
        prior_seconds = previous["seconds"]
        if not 0 <= prior_seconds < 3600:
            raise ValueError("prior attempt has exhausted the experiment budget")
    global_deadline = started+3600-prior_seconds
    manifest = dict(**definitions(), commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        phase_seconds=PHASE_SECONDS, seeds=SEEDS, subsets=SUBSETS, config=asdict(config()),
        started_unix=time.time(), started_clock=started, global_deadline=global_deadline,
        prior_attempt=str(prior_attempt) if prior_attempt else None, prior_seconds=prior_seconds,
        python=sys.version, dependencies={k: importlib.metadata.version(k) for k in ("numpy", "gurobipy", "kaiwu")},
        source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                       for p in list((ROOT / "suc_benders").glob("*.py")) + list((ROOT / "data/rts24").glob("*.json"))})
    save(directory / "manifest.json", manifest)
    for path in manifest["source_sha256"]:
        target = directory / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    for case in CASES:
        n = int(case.split("-")[0][1:])
        save(directory / "inputs" / case / "units.json", case_input(n))
        profile = "stochastic" if case.endswith("stoch") else "deterministic"
        save(directory / "inputs" / case / "scenarios.json", read(ROOT / f"data/rts24/{profile}.json"))
    manifest["input_sha256"] = {p.relative_to(directory).as_posix(): sha(p) for p in (directory / "inputs").rglob("*.json")}
    save(directory / "manifest.json", manifest)
    for phase, seconds in PHASE_SECONDS.items():
        phase_started = time.perf_counter()
        deadline = min(global_deadline, phase_started+seconds)
        save(directory / "progress.json", dict(phase=phase, started=phase_started, deadline=deadline))
        print(f"phase {phase} started; remaining {global_deadline-time.perf_counter():.1f}s", flush=True)
        if phase == "A":
            # Freeze ALL snapshots before ANY integer reference solve.
            for case in CASES:
                supervise(directory, directory / "cases" / case, "init", case, 60, deadline)
            for case in CASES:
                supervise(directory, directory / "references" / case, "reference", case, 30, deadline)
        elif phase == "B":
            for s in manifest["snapshots"]:
                folder = directory / "snapshots" / s["id"]
                for kind, budget in (("native", 12), ("qubo", 16)):
                    supervise(directory, folder / kind, kind, s["id"], budget, deadline)
                for t in (t for t in manifest["trials"] if t["snapshot"] == s["id"]):
                    # Five seconds cumulative SA; the 12-second envelope allows imports/serialization.
                    supervise(directory, directory / "trials" / t["id"], "sample", t["id"], 12, deadline)
                print(f"snapshot {s['id']} finished", flush=True)
        elif phase == "C":
            for d in manifest["runs"]:
                supervise(directory, directory / "end_to_end" / d["id"], "end", d["id"], 90, deadline)
                print(f"end-to-end {d['id']} finished", flush=True)
        else:
            supervise(directory, directory / "report-worker", "report", "all", seconds, deadline)
        save(directory / f"phase-{phase}.json", dict(seconds=time.perf_counter()-phase_started, limit=seconds))
    save(directory / "state.json", dict(status="completed" if (directory / "summary.json").exists() else "report_incomplete",
         seconds=time.perf_counter()-started, total_attempt_seconds=prior_seconds+time.perf_counter()-started,
         within_total_budget=time.perf_counter() <= global_deadline))
    checksum(directory)


def checksum(directory):
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (directory / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(directory).as_posix()}\n" for p in files), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prior-attempt", type=Path, help="debit an archived infrastructure-aborted attempt from the 3600-second cap")
    parser.add_argument("--worker", choices=("init", "reference", "native", "qubo", "sample", "end", "report"), help=argparse.SUPPRESS)
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
        run(args.output_dir, args.prior_attempt.resolve() if args.prior_attempt else None)


if __name__ == "__main__":
    main()
