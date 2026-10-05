"""Deadline-aware quality, attribution, candidate, and physical audit tables."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import time

from .data import load_suc_data
from .rts24_experiment import read_events
from .subagent import publish, sha
from .validation import schedule_object, validate_schedule


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize(events, result, budget, reference_cost):
    evaluations = [e for e in events if e["kind"] == "evaluation" and e["elapsed_seconds"] <= budget]
    feasible = [e for e in evaluations if e["feasible"] and e["objective"] is not None]
    best, curve = None, []
    first_llm, sa_after_llm = None, 0.
    for e in feasible:
        if best is None or e["objective"] < best:
            gain = best-e["objective"] if best is not None else None
            if e["source"] == "llm" and first_llm is None:
                first_llm = e["elapsed_seconds"]
            if e["source"] == "sa" and first_llm is not None:
                sa_after_llm += gain or 0.
            best = e["objective"]
            curve.append(dict(seconds=e["elapsed_seconds"], objective=best, source=e["source"], gain=gain))
    checkpoints = {str(t): min((e["objective"] for e in feasible if e["elapsed_seconds"] <= t), default=None)
                   for t in (300, 600, 900, 1800)}
    thresholds = {str(gap): next((r["seconds"] for r in curve
        if reference_cost and r["objective"] <= reference_cost*(1+gap)), None) for gap in (.01, .005)}
    proposals = defaultdict(list)
    seen = set()
    calls = []
    for e in events:
        if e["kind"] == "evaluation":
            seen.add(e["commitment_hash"])
        entries = [e] if e["kind"] == "proposal" else e.get("diagnostics", {}).get("proposals", []) if e["kind"] in {"llm_end", "rule_end"} else []
        if e["kind"] in {"llm_end", "rule_end"}:
            calls.append(dict(kind=e["kind"], failed=e["diagnostics"].get("failed", False),
                              usage=e["diagnostics"].get("usage"), message=e["diagnostics"].get("message")))
        for p in entries:
            p = dict(p)
            identity = p.get("commitment_id")
            p["duplicate"] = bool(p.get("duplicate") or p.get("outcome") == "duplicate"
                                  or ("block" not in p and identity and identity in seen))
            if identity:
                seen.add(identity)
            proposals[p.get("origin", "llm")].append(p)
    stages = {}
    for origin, rows in proposals.items():
        repaired = [p for p in rows if p.get("repaired_discrete_feasible")]
        distances = [p["hamming_u"] for p in repaired if p.get("hamming_u") is not None]
        ids = {p["commitment_id"] for p in repaired}
        measured = [e for e in evaluations if e["source"] == origin and e["commitment_hash"] in ids]
        valid = [e for e in measured if e["feasible"]]
        stages[origin] = dict(raw_count=len(rows), raw_feasible_count=sum(bool(p.get("raw_discrete_feasible")) for p in rows),
            raw_feasible_rate=sum(bool(p.get("raw_discrete_feasible")) for p in rows)/len(rows),
            repaired_count=len(repaired), repair_bits=distances,
            repair_bits_mean=statistics.mean(distances) if distances else None,
            repair_bits_min=min(distances, default=None), repair_bits_max=max(distances, default=None),
            repaired_duplicates=sum(p["duplicate"] for p in repaired),
            repaired_duplicate_rate=sum(p["duplicate"] for p in repaired)/len(repaired) if repaired else None,
            unique_repaired=len(ids), full_evaluated=len(measured), full_feasible=len(valid),
            full_feasible_rate=len(valid)/len(measured) if measured else None,
            improving_count=sum(e["improved"] for e in valid),
            improvement_rate=sum(e["improved"] for e in valid)/len(measured) if measured else None)
    sa = [e for e in events if e["kind"] == "sa_start"]
    diagnostic = result.get("diagnostics", {})
    return dict(status=result.get("status"), dual_converged=result.get("status") == "DUAL_CONVERGED",
        certified_lower_bound=diagnostic.get("reliable_lower"), primal_gap=diagnostic.get("primal_gap"),
        coordination_upper=diagnostic.get("coordination_upper"), termination_reason=diagnostic.get("termination_reason"),
        objective=best, reference_difference=best/reference_cost-1 if best is not None and reference_cost else None,
        checkpoints=checkpoints, threshold_seconds=thresholds, curve=curve, first_llm_improvement_seconds=first_llm,
        sa_improvement_after_llm=sa_after_llm, candidate_stages=stages, calls=calls,
        llm_started=sum(e["kind"] == "llm_start" for e in events),
        rule_started=sum(e["kind"] == "rule_start" for e in events), sa_calls=len(sa),
        sa_seeds=[e.get("seed") for e in sa], timings=result.get("diagnostics", {}).get("timings", {}),
        time_limit_overshoot=result.get("diagnostics", {}).get("time_limit_overshoot"),
        incomplete_evaluations=sum(e["kind"] == "evaluation_incomplete" for e in events),
        sa_returned_raw=sum(e["raw_candidates"] for e in events if e["kind"] == "sampling"),
        sa_returned_raw_feasible=sum(e["raw_discrete_feasible"] for e in events if e["kind"] == "sampling"),
        sa_returned_after_deadline=sum(e["raw_candidates"] for e in events if e["kind"] == "sampling" and e["elapsed_seconds"] > budget),
        late_evaluations_excluded=sum(e["kind"] == "evaluation" and e["elapsed_seconds"] > budget for e in events))


def report(directory):
    directory = Path(directory)
    started = time.perf_counter()
    manifest = read(directory / "manifest.json")
    reference = read(directory / "reference/result.json")
    cost = reference.get("objective")
    source_ok = all(sha(directory / "snapshot" / p) == h for p, h in manifest["source_sha256"].items())
    data, scenarios = load_suc_data(directory / "snapshot/data/rts24/units.json",
                                    directory / "snapshot/data/rts24/stochastic.json")
    rows, checks = [], []
    for group, seed in manifest["order"]:
        run = directory / f"{group}-{seed}"
        if not run.exists():
            continue
        events, result = read_events(run / "events.jsonl"), read(run / "result.json")
        summary = summarize(events, result, manifest["budget_seconds"], cost)
        validation_errors, checked = [], 0
        for e in events:
            if e["kind"] != "evaluation" or not e["feasible"] or e["elapsed_seconds"] > manifest["budget_seconds"]:
                continue
            try:
                validate_schedule(data, scenarios, schedule_object(e))
                checked += 1
            except (AssertionError, ValueError) as exc:
                validation_errors.append(str(exc))
        state = read(run / "state.json")
        seeds = summary["sa_seeds"]
        expected = [seed+i if group == "B" else seed for i in range(len(seeds))]
        protocol_ok = (seeds == expected and summary["llm_started"] <= (3 if group in {"C", "D"} else 0)
                       and (group != "C" or summary["sa_calls"] == 0)
                       and (group != "E" or manifest["rule_max_calls"] is None or summary["rule_started"] <= manifest["rule_max_calls"]))
        audit = dict(group=group, seed=seed, physical_checked=checked, errors=validation_errors,
            protocol_ok=protocol_ok, result_objective_consistent=(not result or result.get("objective") == summary["objective"]))
        checks.append(audit)
        summary.update(group=group, seed=seed, state=state, audit=audit)
        publish(run / "summary.json", summary)
        rows.append(summary)
    complete = read(directory / "state.json").get("status") == "completed"
    completed_searches = sum(r["state"].get("status") in {"TIME_LIMIT", "LLM_CALL_LIMIT", "EVALUATION_LIMIT", "DUAL_CONVERGED", "INNER_NOT_CONVERGED", "MAX_ITERATIONS"} for r in rows)
    passed = bool(complete and len(rows) == 15 and completed_searches == 15 and source_ok
        and cost and read(directory / "reference/audit.json").get("feasible_solution")
        and all(r["objective"] is not None for r in rows)
        and all(not c["errors"] and c["protocol_ok"] and c["result_objective_consistent"] for c in checks))
    payload = dict(complete=complete, passed=passed, completed_searches=completed_searches,
        source_hashes_match=source_ok, reference_cost=cost, reference_lower_bound=reference.get("lower_bound"),
        llm_started=sum(r["llm_started"] for r in rows), runs=rows, posthoc_audit_seconds=time.perf_counter()-started)
    publish(directory / "summary.json", payload)
    with (directory / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "seed", "seconds", "objective", "source", "gain"])
        writer.writeheader()
        for r in rows:
            writer.writerows(dict(group=r["group"], seed=r["seed"], **point) for point in r["curve"])
    lines = ["# RTS-24 最小必要性证据实验", "",
        f"状态：{'已完成' if complete else '运行中'}；完成 {completed_searches}/15 次搜索；最终验收：{passed}。", "",
        "统一预算 1800 秒，从初始化起计时；种子 7、17、27；单线程；组别顺序预先随机化。",
        "C、D 每次最多 3 次 Astra/max 调用，每次最多 4 个候选；新会话；本地种子不保证模型复现。",
        f"E 总轮数限制：{manifest['rule_max_calls']}（空表示不限）；每次停滞触发最多连续三轮，每轮最多四个修复后不同候选。",
        "规则排序采用与 LLM 相同的前 60 项收益估计；未列小时的燃料变化按零估计，保留已知固定费用并计入启动/停机费用。",
        "失败、截断、等待均计入预算；截止后完成的候选不计分，超时开销单列。", "",
        f"独立参考可行成本：{cost}；下界：{reference.get('lower_bound')}；600 秒、MIPGap=1e-6。",
        "参考成本差不是算法认证最优间隙。参考解未提供给搜索或模型。", "",
        "| 组别 | 种子 | 状态 | 最佳成本 | 参考差 | ≤1% 秒 | ≤0.5% 秒 | SA/LLM/规则次数 | LLM 后 SA 降本 |", "|---|---:|---|---:|---:|---:|---:|---|---:|"]
    def number(x, percent=False):
        return "—" if x is None else f"{100*x:.4f}%" if percent else f"{x:.3f}"
    for r in sorted(rows, key=lambda r: (r["group"], r["seed"])):
        lines.append(f"| {r['group']} | {r['seed']} | {r['state'].get('status')} | {number(r['objective'])} | {number(r['reference_difference'], True)} | {number(r['threshold_seconds']['0.01'])} | {number(r['threshold_seconds']['0.005'])} | {r['sa_calls']}/{r['llm_started']}/{r['rule_started']} | {number(r['sa_improvement_after_llm'])} |")
    lines += ["", "## 算法停止与认证间隙", "",
              "DUAL_CONVERGED 只表示拉格朗日对偶协调收敛；其余正常结束状态分别表示内层/外层上限或资源终止。",
              "| 组别/种子 | 对偶收敛 | 认证下界 | 原问题 gap | 协调上界 |", "|---|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["group"], r["seed"])):
        lines.append(f"| {r['group']}/{r['seed']} | {r['dual_converged']} | {number(r['certified_lower_bound'])} | {number(r['primal_gap'], True)} | {number(r['coordination_upper'])} |")
    lines += ["", "## 固定时刻的最佳成本", "", "C 提前用完额度后，保持最后的已验证解；未达标以 — 表示。", "",
              "| 组别/种子 | 300 秒 | 600 秒 | 900 秒 | 1800 秒 |", "|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["group"], r["seed"])):
        lines.append(f"| {r['group']}/{r['seed']} | " + " | ".join(number(r["checkpoints"][str(t)]) for t in (300, 600, 900, 1800)) + " |")
    lines += ["", "逐次最佳解曲线见 curves.csv；逐级候选统计、修复位数、资源分项及独立物理校验见 summary.json 和各运行的 summary.json。",
        "原始可行率以收到的提案为分母；修复后重复率以成功修复的提案为分母；完整调度可行率和改进率以该来源实际完成的不同完整评估为分母。",
        "规则搜索为获得四个不同候选而尝试的失败/重复修复均记入原始候选与时间。块级可行筛选未送入完整调度的候选不混入完整评估分母。",
        "模型 token 用量未知；总等待计入 LLM 时间。求解期间物理复核计入 validation；本报告的额外事后审计时间另列。",
        "", "三个种子只用于筛查是否值得扩大实验，不构成统计显著性或跨实例优势证据。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload
