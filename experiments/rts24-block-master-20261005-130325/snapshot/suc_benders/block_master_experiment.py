"""Frozen 24/3/6-hour SA master microbenchmark; no LLM or outer-loop claims."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from .block_master_model import (TOL, RecordedQubo, continuous_relaxation, fixed_master_value,
    point_record, priced_blocks, row_record, solve_master)
from .candidates import repair_commitment
from .data import load_suc_data
from .paper_model import IndexedUC
from .quantum import solve_qubo
from .solver import BendersConfig, _construct_feasible_candidate
from .subagent import publish, sha


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 27)
HOURS = (24, 3, 6)
BUDGET = 3600.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def point_id(block, point):
    return hashlib.sha256(bytes(int(point[k]) for k in block.bkeys)).hexdigest()


def relative_gap(value, reference):
    if value is None or not reference.get("certified"):
        return None
    return (value-reference["objective"])/max(1., abs(reference["objective"]))


def native_checks(block, cuts, point, theta):
    discrete = block.discrete_checks(point)
    qlo, _ = block.objective_bounds(block.costs, True)
    optimality = max([0., qlo-theta] + [c.value(point)-theta for c in cuts if c.feasible])
    feasibility = max([0.] + [c.value(point) for c in cuts if not c.feasible])
    return dict(feasible=discrete["feasible"] and optimality <= TOL and feasibility <= TOL,
                optimality_violation=optimality, feasibility_cut_violation=feasibility, discrete=discrete)


def repair_sample(block, point, initial):
    """Original full-horizon repair; all unobserved hours use the common seed."""
    data, scenarios = block.uc.data, block.uc.scenarios
    raw = initial.u.copy()
    for k, value in point.items():
        if k[0] == "u":
            raw[k[1], k[2]] = value
    fixed = repair_commitment(data, scenarios, raw)
    if fixed is None:
        return None, None, None
    changed = fixed.u != raw
    local = int(changed[:, block.times].sum())
    outside = int(changed.sum())-local
    projected = {k: int(getattr(fixed, k[0])[k[1], k[2]]) for k in block.bkeys}
    return projected, local, outside


def evaluate_sample(master, block, cuts, bits, initial, reference, seed, index):
    point, theta = master.decode(bits)
    native = native_checks(block, cuts, point, theta)
    encoded = master.encoded_checks(bits)
    full = native["feasible"] and encoded["feasible"]
    objective = theta + sum(block.costs[k]*point[k] for k in block.bkeys) + block.constant
    phase = time.perf_counter()
    raw_fixed = fixed_master_value(block, cuts, point)
    fixed_raw_seconds = time.perf_counter()-phase
    phase = time.perf_counter()
    repaired, changes, outside = repair_sample(block, point, initial)
    repair_seconds = time.perf_counter()-phase
    phase = time.perf_counter()
    repaired_value = fixed_master_value(block, cuts, repaired) if repaired is not None else None
    fixed_repaired_seconds = time.perf_counter()-phase
    return dict(seed=seed, sample=index, raw_point=point_record(point), raw_id=point_id(block, point),
        decoded_theta=float(theta), raw_discrete_feasible=bool(native["discrete"]["feasible"]),
        raw_native_master_feasible=bool(native["feasible"]), raw_encoded_feasible=bool(encoded["feasible"]),
        full_master_feasible=bool(full), native_checks=native, encoded_checks=encoded,
        decoded_objective=float(objective), raw_objective=float(objective) if full else None,
        raw_gap=relative_gap(objective, reference) if full else None,
        classical_fixed_raw_value=raw_fixed, classical_fixed_raw_gap=relative_gap(raw_fixed, reference),
        repaired_id=point_id(block, repaired) if repaired is not None else None,
        repaired_point=point_record(repaired), repair_bits=changes,
        repair_fraction=changes/(block.uc.data.n_units*len(block.times)) if changes is not None else None,
        repair_outside_block_bits=outside, repaired_value=repaired_value,
        repaired_gap=relative_gap(repaired_value, reference),
        repair_objective_change=repaired_value-raw_fixed if repaired_value is not None and raw_fixed is not None else None,
        repair_seconds=repair_seconds, fixed_raw_seconds=fixed_raw_seconds,
        fixed_repaired_seconds=fixed_repaired_seconds)


def snapshot_spec(block, cuts, lower, upper, config):
    return dict(times=list(block.times), variables=[dict(key=list(k), bounds=list(block.uc.bounds[k]),
        linear=block.costs[k], quadratic=block.uc.quadratic.get(k, 0.)) for k in block.keys],
        rows=[row_record(r) for r in block.rows], cuts=[dict(coefficients=point_record(c.a),
        constant=c.c, feasible=c.feasible) for c in cuts], fixed=point_record(block.fixed),
        objective_constant=block.constant, physical_cost_scale=block.scale,
        lower=lower, upper=upper, penalty=config.penalty, theta_step=config.theta_step,
        reserve_scale=config.reserve_scale)


def reference_record(reference):
    return {**reference, "point": point_record(reference.get("point"))}


def classical_trace(block, initial, deadline, progress):
    costs = block.costs
    point = {k: int(getattr(initial, k[0])[k[1], k[2]]) for k in block.bkeys}
    phase = time.perf_counter()
    first = block.dispatch(point, costs, min(deadline, time.perf_counter()+60))
    if first is None or not first["feasible"]:
        return {}, [], "initial_dispatch_unavailable"
    cuts = [first["cut"]]
    cut_ids = {digest(dict(a=point_record(first["cut"].a), c=first["cut"].c, feasible=True))}
    upper = first["cost"]
    upper_point = dict(point)
    states, steps = {}, []
    status = "five_iterations"
    for iteration in range(1, 6):
        if time.perf_counter() >= deadline:
            status = "budget_exhausted"
            break
        progress(iteration)
        lp = solve_master(block, cuts, deadline, integral=False)
        if not lp["certified"] or lp["objective"] is None:
            status = "lp_bound_unavailable"
            break
        lower = lp["objective"]-block.constant-1e-5
        # Freeze bounds BEFORE the current integer oracle. Do not tighten to its optimum.
        reference = solve_master(block, cuts, deadline)
        snapshot = dict(cuts=list(cuts), lower=lower, upper=upper, reference=reference,
                        iteration=iteration, lp=lp, upper_point=dict(upper_point))
        if iteration == 1:
            states["initial"] = snapshot
        step = dict(iteration=iteration, cut_count=len(cuts), upper_before=upper,
                    lp=reference_record(lp), reference=reference_record(reference))
        steps.append(step)
        if iteration == 5:
            states["final"] = snapshot
        if reference["point"] is None:
            status = "oracle_no_solution"
            break
        if not block.discrete_checks(reference["point"])["feasible"]:
            raise ValueError("native master oracle returned a discrete-infeasible point")
        dispatch = block.dispatch(reference["point"], costs, min(deadline, time.perf_counter()+60))
        if dispatch is None:
            status = "dispatch_unavailable"
            break
        if dispatch["feasible"] and dispatch["cost"] < upper:
            upper = dispatch["cost"]
            upper_point = dict(reference["point"])
        step.update(dispatch_feasible=dispatch["feasible"], upper_after=upper)
        bound = reference.get("lower_bound")
        if bound is not None and upper+block.constant-bound <= 1e-6*max(1., abs(upper+block.constant)):
            states["final"] = snapshot
            status = "converged"
            break
        cut = dispatch["cut"]
        cut_id = digest(dict(a=point_record(cut.a), c=cut.c, feasible=cut.feasible))
        if cut_id not in cut_ids:
            cuts.append(cut)
            cut_ids.add(cut_id)
    for state in states.values():
        state["trace_seconds"] = time.perf_counter()-phase
    return states, steps, status


def audit_encoding(master, block, snapshot):
    """Check a pre-existing feasible witness; NEVER pass these bits to SA."""
    witness = snapshot["upper_point"]
    values = {n: 0 for n in master.names}
    from .paper_model import name, key
    for k in master.vars:
        values[name(k)] = witness[k]
    for n in master.zvars:
        values[n] = 1  # ceil(existing feasible upper bound), not current oracle optimum
    for n in master.names:
        if n.startswith("and_"):
            _, i, t = n.split("_")
            values[n] = witness[key("v", int(i), int(t))]*witness[key("w", int(i), int(t)+1)]
    for row in master.inequalities:
        target = -row["original_constant"]-sum(v*values[k] for k,v in row["original_coefficients"].items())
        if target < 0:
            raise ValueError(f"existing feasible witness violates encoded {row['label']}")
        slack = [(k,v) for k,v in row["coefficients"].items() if k not in row["original_coefficients"]]
        for k, weight in sorted(slack, key=lambda item: item[1], reverse=True):
            if weight <= target:
                values[k] = 1
                target -= weight
        if target != 0:
            raise ValueError(f"slack cannot represent witness at {row['label']}")
    bits = np.asarray([values[n] for n in master.names], dtype=np.uint8)
    encoded = master.encoded_checks(bits)
    point, theta = master.decode(bits)
    native = native_checks(block, snapshot["cuts"], point, theta)
    if not encoded["feasible"] or not native["feasible"]:
        raise ValueError("existing feasible witness failed native/encoded audit")
    return dict(passed=True, witness_source="incumbent before current integer oracle",
                not_used_as_sa_initial_state=True, native=native, encoded=encoded)


def benchmark_snapshot(directory, block, snapshot, initial, config, deadline):
    phase = time.perf_counter()
    master = RecordedQubo(block, block.costs, snapshot["cuts"], snapshot["lower"], snapshot["upper"], config)
    build_seconds = time.perf_counter()-phase
    publish(directory / "encoding-audit.json", audit_encoding(master, block, snapshot))
    matrix = master.matrix
    nonzero = np.abs(matrix[matrix != 0])
    np.savez_compressed(directory / "qubo.npz", matrix=matrix, offset=master.model.get_offset(),
                        variable_names=np.asarray(master.names))
    metadata = dict(bits=len(master.names), binary_decision_bits=len(master.vars),
        objective_bits=len(master.zvars), auxiliary_bits=len(master.names)-len(master.vars)-len(master.zvars),
        coefficient_percentiles={str(p): float(np.percentile(nonzero, p)) for p in (0, 1, 50, 99, 100)} if len(nonzero) else {},
        objective_energy_unit=master.energy_unit,
        normalized_energy_unit=master.energy_unit/max(1., float(nonzero.max()) if len(nonzero) else 1.),
        build_seconds=build_seconds, model_offset=float(master.model.get_offset()),
        inequalities=master.inequalities, variables=master.names, completed_seeds=[])
    publish(directory / "encoding.json", metadata)
    samples, calls, pending = [], [], list(SEEDS)
    for seed in SEEDS:
        if time.perf_counter() >= deadline:
            break
        phase = time.perf_counter()
        bits_list = solve_qubo(matrix, replace(config.quantum, kaiwu_seed=seed), energy_unit=master.energy_unit)
        seconds = time.perf_counter()-phase
        late = time.perf_counter() >= deadline
        np.savez_compressed(directory / f"samples-{seed}.npz", samples=np.asarray(bits_list, dtype=np.uint8))
        calls.append(dict(seed=seed, seconds=seconds, returned=len(bits_list), late=late))
        if late:
            break
        for index, bits in enumerate(bits_list):
            if time.perf_counter() >= deadline:
                break
            sample = evaluate_sample(master, block, snapshot["cuts"], bits, initial, snapshot["reference"], seed, index)
            if time.perf_counter() >= deadline:
                break
            samples.append(sample)
        if sum(s["seed"] == seed for s in samples) == len(bits_list):
            pending.remove(seed)
            metadata["completed_seeds"].append(seed)
        else:
            break
    publish(directory / "encoding.json", metadata)
    payload = dict(samples=samples, calls=calls, pending_seeds=pending, complete=not pending,
                   build_seconds=build_seconds, repair_seconds=sum(s["repair_seconds"] for s in samples))
    publish(directory / "samples.json", payload)
    return payload


def aggregate(rows):
    samples = [s for row in rows for s in row.get("samples", {}).get("samples", [])]
    repaired = [s for s in samples if s["repaired_id"] is not None]
    duplicates = 0
    raw_unique = repaired_unique = 0
    for row in rows:
        ss = row.get("samples", {}).get("samples", [])
        raw_unique += len({s["raw_id"] for s in ss})
        rr = [s["repaired_id"] for s in ss if s["repaired_id"] is not None]
        repaired_unique += len(set(rr))
        duplicates += len(rr)-len(set(rr))
    certified = [r for r in rows if r["reference"]["certified"]]
    successful = sum(any(s["classical_fixed_raw_gap"] is not None and s["classical_fixed_raw_gap"] <= .05
                        for s in r.get("samples", {}).get("samples", [])) for r in certified)
    return dict(snapshots=len(rows), certified=len(certified), candidates=len(samples),
        raw_discrete_feasible=sum(s["raw_discrete_feasible"] for s in samples),
        raw_discrete_feasible_rate=sum(s["raw_discrete_feasible"] for s in samples)/len(samples) if samples else None,
        full_master_feasible=sum(s["full_master_feasible"] for s in samples),
        raw_unique=raw_unique, repaired_unique=repaired_unique,
        repair_fraction=sum(s["repair_fraction"] for s in repaired)/len(repaired) if repaired else None,
        repaired_duplicate_rate=duplicates/len(repaired) if repaired else None,
        certified_with_5pct_unmodified_commitment=successful,
        success_rate=successful/len(certified) if certified else None,
        build_seconds=sum(r.get("encoding", {}).get("build_seconds", 0.) for r in rows),
        annealing_seconds=sum(c["seconds"] for r in rows for c in r.get("samples", {}).get("calls", [])),
        repair_seconds=sum(s["repair_seconds"] for s in samples),
        fixed_objective_seconds=sum(s["fixed_raw_seconds"]+s["fixed_repaired_seconds"] for s in samples),
        sampling_complete=bool(rows) and all(r.get("samples", {}).get("complete") for r in rows))


def report(directory):
    import csv
    manifest = read(directory / "manifest.json")
    rows = []
    for folder in sorted((directory / "snapshots").glob("*")):
        metadata = read(folder / "snapshot.json")
        if not metadata:
            continue
        metadata.update(id=folder.name, reference=read(folder / "reference.json"),
            samples=read(folder / "samples.json"), encoding=read(folder / "encoding.json"))
        rows.append(metadata)
    traces = [read(directory / "traces" / f"{t['id']}.json") for t in manifest["traces"]]
    groups = {}
    for hours in HOURS:
        own = [r for r in rows if r["hours"] == hours]
        resolved = set()
        unresolved = 0
        for definition, trace in zip(manifest["traces"], traces):
            if definition["hours"] != hours:
                continue
            for role in ("initial", "final"):
                identity = trace.get("slots", {}).get(role)
                if identity:
                    resolved.add(identity)
                else:
                    unresolved += 1
        group = aggregate(own)
        group["planned_unique_or_unresolved"] = len(resolved)+unresolved
        group["reference_coverage"] = group["certified"]/max(1, len(resolved)+unresolved)
        group["unresolved_slots"] = unresolved
        group["classical_seconds"] = sum(t.get("classical_seconds",0.) for d,t in zip(manifest["traces"],traces) if d["hours"]==hours)
        groups[str(hours)] = group
    baseline = groups["24"]
    for hours in (3, 6):
        g = groups[str(hours)]
        reductions = all(g[k] is not None and baseline[k] is not None and g[k] < baseline[k]
                         for k in ("repair_fraction", "repaired_duplicate_rate"))
        g["screening_passed"] = bool(g["sampling_complete"] and baseline["sampling_complete"]
            and g["unresolved_slots"] == 0 and baseline["unresolved_slots"] == 0 and g["reference_coverage"] >= .8
            and g["raw_discrete_feasible_rate"] is not None and g["raw_discrete_feasible_rate"] >= .5
            and g["success_rate"] is not None and g["success_rate"] >= .8 and reductions)
        g["repair_and_duplicate_reduced"] = reductions
    state = read(directory / "state.json")
    complete = (state.get("status") == "completed" and all(traces)
        and all(t.get("status") in {"converged", "five_iterations"} for t in traces)
        and all(r["samples"].get("complete") for r in rows)
        and all(groups[str(h)]["unresolved_slots"] == 0 for h in HOURS))
    by_price = {}
    price_ids = {}
    for h in HOURS:
        for p in (("zero",) if h==24 else ("zero","qp")):
            ids = {identity for definition,trace in zip(manifest["traces"],traces)
                   if definition["hours"]==h and definition["price"]==p
                   for identity in trace.get("slots",{}).values()}
            price_ids[f"{h}h-{p}"] = ids
            by_price[f"{h}h-{p}"] = aggregate([r for r in rows if r["id"] in ids])
    payload = dict(complete=complete, state=state, groups=groups, by_price=by_price,
        snapshots=len(rows), sa_calls=sum(len(r["samples"].get("calls", [])) for r in rows),
        candidate_count=sum(g["candidates"] for g in groups.values()),
        uncertified_snapshots=[r["id"] for r in rows if not r["reference"].get("certified")],
        incomplete_traces=[t["id"] for t, result in zip(manifest["traces"], traces)
                           if result.get("status") not in {"converged", "five_iterations"}],
        incomplete_snapshots=[r["id"] for r in rows if not r["samples"].get("complete")],
        snapshot_rows=[{k:v for k,v in r.items() if k not in {"spec", "samples", "encoding"}} for r in rows])
    publish(directory / "summary.json", payload)
    fields = ["snapshot", "hours", "seed", "sample", "raw_discrete_feasible", "full_master_feasible", "raw_gap",
        "classical_fixed_raw_gap", "repair_bits", "repair_fraction", "repair_outside_block_bits", "repaired_gap"]
    with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for sample in row["samples"].get("samples", []):
                writer.writerow(dict(snapshot=row["id"], hours=row["hours"], **{k:sample[k] for k in fields[2:]}))
    with (directory / "snapshot-summary.csv").open("w",encoding="utf-8",newline="") as stream:
        fields2=["snapshot","hours","price","iteration","bits","reference_certified","reference_objective",
                 "reference_bound","raw_discrete_rate","full_master_count","best_raw_gap","best_fixed_raw_gap","best_repaired_gap"]
        writer=csv.DictWriter(stream,fieldnames=fields2)
        writer.writeheader()
        for row in rows:
            ss=row["samples"].get("samples",[])
            gaps={k:min((s[k] for s in ss if s[k] is not None),default=None)
                  for k in ("raw_gap","classical_fixed_raw_gap","repaired_gap")}
            writer.writerow(dict(snapshot=row["id"],hours=row["hours"],price=row["price"],iteration=row["iteration"],
                bits=row["encoding"].get("bits"),reference_certified=row["reference"].get("certified"),
                reference_objective=row["reference"].get("objective"),reference_bound=row["reference"].get("lower_bound"),
                raw_discrete_rate=sum(s["raw_discrete_feasible"] for s in ss)/len(ss) if ss else None,
                full_master_count=sum(s["full_master_feasible"] for s in ss),best_raw_gap=gaps["raw_gap"],
                best_fixed_raw_gap=gaps["classical_fixed_raw_gap"],best_repaired_gap=gaps["repaired_gap"]))
    pct = lambda v: "—" if v is None else f"{100*v:.2f}%"
    lines = ["# 经典价格下的分块 SA 主问题微基准", "", f"全部预定工作完成：{complete}；冻结快照 {len(rows)} 个；SA 调用 {payload['sa_calls']} 次。", "",
        "本实验固定经典乘子与 Benders 主问题，比较 24/3/6 小时块。没有 LLM，没有完整外层协调或全天可行解恢复，不构成全天收敛、最终成本或加速证明。",
        "保留现有罚系数、量化和 SA 参数。每快照每种子只采样一次，种子为 7/17/27，候选池 16。经典参考与 SA 使用同一个原生主问题，编码边界在当前整数参考求解前冻结。", "",
        "| 块长 | 快照 | 认证覆盖 | 原始离散可行率 | 完整主问题可行数 | 平均修复位比例 | 修复后重复率 | ≤5% 未改开停机命中 | 门槛通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for h in HOURS:
        g = groups[str(h)]
        lines.append(f"| {h}h | {g['snapshots']} | {pct(g['reference_coverage'])} | {pct(g['raw_discrete_feasible_rate'])} | {g['full_master_feasible']}/{g['candidates']} | {pct(g['repair_fraction'])} | {pct(g['repaired_duplicate_rate'])} | {g['certified_with_5pct_unmodified_commitment']}/{g['certified']} | {g.get('screening_passed','对照')} |")
    lines += ["", "## 按乘子状态分层", "", "| 组别 | 样本 | 原始离散可行率 | 完整主问题可行数 | ≤5% 未改开停机命中 |",
              "|---|---:|---:|---:|---:|"]
    for label,g in by_price.items():
        lines.append(f"| {label} | {g['candidates']} | {pct(g['raw_discrete_feasible_rate'])} | {g['full_master_feasible']} | {g['certified_with_5pct_unmodified_commitment']}/{g['certified']} |")
    lines += ["", "## 分项时间（秒）", "", "| 块长 | 经典轨迹与参考 | QUBO 构建 | SA | 原规则修复 | 固定开停机目标重算 |", "|---|---:|---:|---:|---:|---:|"]
    for h in HOURS:
        g=groups[str(h)]
        lines.append(f"| {h}h | " + " | ".join(f"{g[k]:.3f}" for k in ("classical_seconds","build_seconds","annealing_seconds","repair_seconds","fixed_objective_seconds"))+" |")
    lines += ["", "## 口径与边界", "",
        "原始离散可行性含块内启停、最短时间及容量；完整主问题还要求原生割和所有编码辅助关系成立。量化约束残差另外保存，不能用量化可行冒充原生可行。",
        "classical_fixed_raw_* 是固定全部 u/v/w 后重新计算最小可行目标变量的经典后处理，绝不是 SA 原始目标。gap 分母为 max(1, abs(参考目标))，同时保留绝对目标值和常数；未认证参考的 gap 留空。",
        "修复沿用原全时域算法：非本块小时填入共同初始方案，再修复并投影回块；块外改动单列。修复后目标同样经过目标变量重算。不同块或乘子状态不混用缓存与割。",
        "重复率在每个快照的三个种子候选合集中计算；修复距离按本块开停机位数归一化。结果按快照与价格分别保存，不把重复样本视作独立统计重复。",
        "乘子状态及时间块改变了主问题，跨组结果是可解性筛查，不是相同数学问题上的严格配对；两种经典后处理均明确分栏。",
        "至少 80% 参考认证覆盖、原始离散可行率 50%、至少 80% 认证快照找到 ≤5% 未改开停机候选、修复位比例及重复率同时下降，才通过工程筛查。通过不自动启动下一阶段。", "",
        f"未认证快照：{payload['uncertified_snapshots']}。未完成轨迹：{payload['incomplete_traces']}。未完成采样快照：{payload['incomplete_snapshots']}。", "",
        "逐样本数据见 samples.csv 与 snapshots/*/samples.json；QUBO 矩阵、变量、约束、割、原生参考和原始比特样本均保存在对应快照目录。", "",
        "![可行率与分层目标差](quality.png)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if complete or state.get("status") in {"failed","budget_exhausted","hard_timeout"}:
        plot_results(directory,rows,by_price,price_ids)
    return payload


def plot_results(directory,rows,groups,price_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=list(groups)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    rates=[100*groups[label]["raw_discrete_feasible_rate"] if groups[label]["raw_discrete_feasible_rate"] is not None else np.nan for label in labels]
    axes[0].bar(range(len(labels)),rates,color="#3478aa")
    axes[0].set(ylim=(0,105),ylabel="Raw discrete feasible (%)",title="Before repair")
    for i,value in enumerate(rates):
        axes[0].text(i,3 if not np.isfinite(value) else max(3,value+2),"N/A" if not np.isfinite(value) else f"{value:.1f}",ha="center",fontsize=9)
    for field,color,offset,label in [("raw_gap","#3478aa",-.18,"Raw (fully feasible)"),
        ("classical_fixed_raw_gap","#e49321",0.,"Fixed u/v/w, classical theta"),
        ("repaired_gap","#298747",.18,"Repaired + classical theta")]:
        xs,ys=[],[]
        for i,group in enumerate(labels):
            relevant=[r for r in rows if r["id"] in price_ids[group]]
            values=[min((s[field] for s in r["samples"].get("samples",[]) if s[field] is not None),default=None) for r in relevant]
            finite=[v for v in values if v is not None]
            for jitter,value in zip(np.linspace(-.05,.05,len(finite)),finite):
                xs.append(i+offset+jitter);ys.append(100*value)
        axes[1].scatter(xs,ys,color=color,label=label,s=24,alpha=.75)
    axes[1].axhline(5,color="#555555",linestyle="--",linewidth=1)
    axes[1].set(ylabel="Best native master gap per snapshot (%)",yscale="symlog",title="Missing markers: no eligible candidate")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(labels)),labels,rotation=20)
        ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "quality.png",dpi=170)
    plt.close(fig)


def worker(directory, deadline):
    config = BendersConfig()
    data, scenarios = load_suc_data(ROOT / "data/rts24/units.json", ROOT / "data/rts24/stochastic.json")
    indexed = IndexedUC(data, scenarios)
    initial = _construct_feasible_candidate(data, scenarios)
    publish(directory / "initial.json", dict(u=initial.u.tolist(), v=initial.v.tolist(), w=initial.w.tolist()))
    pricing = continuous_relaxation(indexed, deadline)
    publish(directory / "pricing.json", {**pricing, "point": point_record(pricing["point"]),
        "row_orientation": "a*x+c<=0 (or equality); lambda=-Gurobi Pi", "rows":[row_record(r) for r in indexed.rows]})
    manifest = read(directory / "manifest.json")
    registry = {}
    try:
        for definition in manifest["traces"]:
            if time.perf_counter() >= deadline:
                break
            hours, j, price = definition["hours"], definition["block"], definition["price"]
            multipliers = [0.]*len(indexed.rows) if price == "zero" else pricing["multipliers"]
            blocks, cross = priced_blocks(indexed, hours, multipliers)
            block = blocks[j]
            def progress(iteration):
                publish(directory / "progress.json", dict(trace=definition["id"], phase="classical", iteration=iteration,
                    remaining_seconds=deadline-time.perf_counter(), finished_snapshots=len(registry)))
            phase = time.perf_counter()
            states, steps, status = classical_trace(block, initial, deadline, progress)
            record = dict(**definition, status=status, slots={}, steps=steps,
                classical_seconds=time.perf_counter()-phase,
                cross_rows=cross, lagrangian_constant=block.constant,
                incident_prices=[[i,multipliers[i]] for i in cross if any(k[2] in block.times for k in indexed.rows[i].a)])
            for role, state in states.items():
                spec = snapshot_spec(block, state["cuts"], state["lower"], state["upper"], config)
                identity = digest(spec)
                if identity in registry:
                    record["slots"][role] = registry[identity]
                    continue
                if len(registry) >= 50:
                    raise ValueError("prespecified snapshot ceiling exceeded")
                identifier = f"{definition['id']}-k{state['iteration']}"
                folder = directory / "snapshots" / identifier
                folder.mkdir(parents=True)
                publish(folder / "snapshot.json", dict(id=identifier, hours=hours, block=j, price=price,
                    iteration=state["iteration"], identity=identity, spec=spec, incident_prices=record["incident_prices"],
                    trace_seconds=state["trace_seconds"]))
                publish(folder / "reference.json", reference_record(state["reference"]))
                registry[identity] = identifier
                record["slots"][role] = identifier
                if time.perf_counter() < deadline:
                    publish(directory / "progress.json", dict(trace=definition["id"], phase="SA", snapshot=identifier,
                        remaining_seconds=deadline-time.perf_counter(), finished_snapshots=len(registry)-1))
                    benchmark_snapshot(folder, block, state, initial, config, deadline)
            publish(directory / "traces" / f"{definition['id']}.json", record)
            print(json.dumps(dict(trace=definition["id"], status=status, snapshots=record["slots"],
                remaining_seconds=round(deadline-time.perf_counter(),1))), flush=True)
            report(directory)
        status = "completed" if time.perf_counter() < deadline else "budget_exhausted"
        publish(directory / "state.json", dict(status=status, finished_at=time.time(), overrun=max(0.,time.perf_counter()-deadline)))
    except BaseException as exc:
        publish(directory / "state.json", dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report(directory)


def freeze(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ("snapshot", "snapshots", "traces"):
        (directory / folder).mkdir()
    hashes = {}
    for source in [*ROOT.glob("suc_benders/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("data/rts24/*.json"), ROOT / "requirements.txt"]:
        relative = source.relative_to(ROOT)
        target = directory / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = sha(target)
    traces = [dict(id=f"h{h:02d}-b{j:02d}-{price}", hours=h, block=j, price=price)
              for h in (3, 6) for j in range(24//h) for price in ("zero", "qp")]
    random.Random(20261005).shuffle(traces)
    traces.insert(0, dict(id="h24-b00-zero", hours=24, block=0, price="zero"))
    manifest = dict(created_at=time.time(), budget_seconds=BUDGET, seeds=SEEDS,
        max_snapshots=50, max_sa_calls=150, traces=traces, shuffle_seed=20261005,
        classical_time_limit=60, mip_gap=1e-6, config=asdict(BendersConfig()),
        source_sha256=hashes, executable=sys.executable,
        packages={n:importlib.metadata.version(n) for n in ("numpy","scipy","gurobipy","kaiwu","matplotlib")},
        protocol=dict(no_llm=True, no_outer_loop=True, no_parameter_tuning=True, quantum_candidates=16,
            reference_not_sa_start=True, fixed_prices=["zero","continuous_QP_duals"],
            repair="original full-horizon repair, outside block filled with common initial u",
            slots="first and fifth native master, or final distinct master at earlier convergence; identical specs deduplicated",
            postprocessing="fixed u/v/w objective recomputation reported separately",
            tolerances=dict(native=1e-6, quantized_integer_residual=0, conservative_cut_error=1.)))
    publish(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-clock", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if args.worker:
        try:
            worker(directory, args.deadline_clock)
        except BaseException as exc:
            if read(directory / "state.json").get("status") != "failed":
                publish(directory / "state.json",dict(status="failed",error_type=type(exc).__name__,message=str(exc)))
                report(directory)
            raise
        return
    import os
    manifest = freeze(directory)
    env = os.environ.copy()
    env.update({k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    with (directory / "preflight.log").open("w", encoding="utf-8") as log:
        check = subprocess.run([sys.executable,"-m","unittest","tests.test_block_master_experiment","tests.test_qubo_encoding","tests.test_paper","tests.test_quantum","-v"],
            cwd=directory / "snapshot", env=env, stdout=log, stderr=subprocess.STDOUT)
    if check.returncode:
        publish(directory / "state.json", dict(status="preflight_failed", returncode=check.returncode))
        raise RuntimeError("correctness preflight failed; benchmark not started")
    started = time.perf_counter()
    publish(directory / "state.json", dict(status="running", started_at=time.time(), deadline_clock=started+BUDGET))
    command = [sys.executable,"-u","-m","suc_benders.block_master_experiment","--worker","--output-dir",str(directory),"--deadline-clock",str(started+BUDGET)]
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=directory / "snapshot", env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            process.wait(timeout=BUDGET+30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            publish(directory / "state.json", dict(status="hard_timeout", elapsed_seconds=time.perf_counter()-started))
    report(directory)
    changed = [p for p,h in manifest["source_sha256"].items() if sha(directory / "snapshot" / p) != h]
    publish(directory / "source-verification.json", dict(unchanged=not changed, mismatches=changed,
        worker_exitcode=process.returncode, elapsed_seconds=time.perf_counter()-started))
    files = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256SUMS.txt")
    (directory / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(directory).as_posix()}\n" for p in files),encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), summary=str(directory / "summary.json"), exitcode=process.returncode)))


if __name__ == "__main__":
    main()
