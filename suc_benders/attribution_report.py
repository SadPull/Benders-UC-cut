"""Post-run statistics and figures; no reference data flows back to the solver."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .attribution import CASES, SEEDS, STRATEGIES, now, write_json


CASE_LABELS = {"canonical": "原始负荷", "light_90": "负荷降低一成", "peak_shift_2h": "负荷后移两小时"}
STRATEGY_LABELS = {
    "q0_l0_d0": "仅初始方案", "q0_l0_d1": "关机搜索＋修复",
    "q0_l1_d0": "大模型＋修复", "q0_l1_d1": "大模型＋关机搜索＋修复",
    "q1_l0_d0": "退火＋修复", "q1_l0_d1": "退火＋关机搜索＋修复",
    "q1_l1_d0": "退火＋大模型＋修复", "q1_l1_d1": "退火＋大模型＋关机搜索＋修复",
    "q_only_raw": "原始退火（不修复）", "random_repair": "随机候选＋修复",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint(events, axis, budget):
    """Right-continuous incumbent, held after a normal stop; never look ahead."""
    field = "evaluation_count" if axis == "evaluations" else "elapsed_seconds"
    values = [row["incumbent"] for row in events if not row["cache_hit"]
              and row[field] <= budget and row["incumbent"] is not None]
    return values[-1] if values else None


def describe(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return {"n": int(values.size), "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "std": float(np.std(values, ddof=1)) if values.size > 1 else None}


def paired_effects(costs, *, bootstrap_samples=10000, seed=20260930):
    """Each row is a matched repeat, with columns C00, C10, C01, C11."""
    costs = np.asarray(costs, dtype=float).reshape(-1, 4)
    costs = costs[np.all(np.isfinite(costs), axis=1)]
    contrasts = {
        "q_gain_without_l": [1, -1, 0, 0], "q_gain_with_l": [0, 0, 1, -1],
        "l_gain_without_q": [1, 0, -1, 0], "l_gain_with_q": [0, 1, 0, -1],
        "interaction": [-1, 1, 1, -1],
    }
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(costs), size=(bootstrap_samples, len(costs))) if len(costs) > 1 else None
    rows = []
    for name, coefficients in contrasts.items():
        values = costs @ np.asarray(coefficients)
        row = {"effect": name, **describe(values)}
        if samples is not None:
            lo, hi = np.quantile(values[samples].mean(axis=1), [.025, .975])
            row.update(ci_low=float(lo), ci_high=float(hi))
        else:
            row.update(ci_low=None, ci_high=None)
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def paired_difference(before, after):
    pairs = [(a, b) for a, b in zip(before, after)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    values = np.asarray([a - b for a, b in pairs])
    result = describe(values)
    if len(values) > 1:
        rng = np.random.default_rng(20260930)
        samples = rng.integers(0, len(values), size=(10000, len(values)))
        low, high = np.quantile(values[samples].mean(axis=1), [.025, .975])
        result.update(ci_low=float(low), ci_high=float(high))
    else:
        result.update(ci_low=None, ci_high=None)
    return result


def run_metrics(record, events, calls, sampling, proposals, reference):
    result = record["result"]
    objective, lower_bound = result["objective"], reference["lower_bound"]
    fresh = [row for row in events if not row["cache_hit"]]
    row = {
        "case": record["case"], "seed": record["seed"], "strategy": record["strategy"]["name"],
        "status": result["status"], "objective": objective, "runtime": result["runtime"],
        "feasible": record["validation"]["feasible_solution"], "evaluations": len(fresh),
        "iterations": len(result["history"]), "budget_exhausted": result["status"] == "EVALUATION_LIMIT",
        "scenario_solves": sum(event["scenario_solves"] for event in events),
        "feasible_evaluations": sum(event["feasible"] for event in fresh),
        "cache_hits": sum(event["cache_hit"] for event in events),
        "relative_error_pct": 100 * (objective - reference["objective"]) / reference["objective"]
            if objective is not None and reference["objective"] not in (None, 0) else None,
        "reference_gap_pct": 100 * (objective - lower_bound) / max(1, abs(objective))
            if objective is not None and lower_bound is not None else None,
        "llm_calls": len(calls), "llm_requests": sum(len(call.get("attempts", [])) for call in calls),
        "llm_failed_calls": sum(bool(call["failed"]) for call in calls),
        "llm_successful_calls": sum(not call["failed"] for call in calls),
        "llm_raw_candidates": sum(call.get("raw_count", 0) for call in calls),
        "llm_valid_candidates": sum(call.get("valid_count", 0) for call in calls),
        "llm_duplicate_proposals": sum(p.get("outcome") == "duplicate" for c in calls for p in c.get("proposals", [])),
        "llm_truncated_calls": sum(call.get("message") == "truncated" for call in calls),
        "prompt_tokens_known": 0, "completion_tokens_known": 0, "usage_reported_calls": 0,
        "q_raw_candidates": sum(item["raw_candidates"] for item in sampling),
        "q_raw_feasible": sum(item["raw_discrete_feasible"] for item in sampling),
        "q_unique_u_per_round": float(np.mean([item["unique_u"] for item in sampling])) if sampling else None,
        "max_qubo_bits": max((item["bits"] for item in sampling), default=0),
    }
    if row["q_raw_candidates"]:
        row["q_raw_feasible_rate"] = row["q_raw_feasible"] / row["q_raw_candidates"]
    else:
        row["q_raw_feasible_rate"] = None
    repair_distances = [p["hamming_u"] for p in proposals if p.get("origin") == "quantum" and p.get("hamming_u") is not None]
    row["q_repair_hamming_u_mean"] = float(np.mean(repair_distances)) if repair_distances else None
    for call in calls:
        usage = call.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion = usage.get("completion_tokens", usage.get("output_tokens"))
            if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
                row["usage_reported_calls"] += 1
                row["prompt_tokens_known"] += prompt
                row["completion_tokens_known"] += completion
    row["usage_unknown_calls"] = len(calls) - row["usage_reported_calls"]
    for source in ("quantum", "llm", "decommit", "random", "fallback"):
        candidates = [event for event in fresh if event["origin"] == source]
        row[f"{source}_evaluated"] = len(candidates)
        row[f"{source}_improvements"] = sum(event["improved"] and event["improvement"] is not None for event in candidates)
        row[f"{source}_cost_improvement"] = sum(event["improvement"] or 0.0 for event in candidates)
    for name, value in record["diagnostics"]["timings"].items():
        row[f"seconds_{name}"] = value
    if objective is not None:
        best = min((event for event in fresh if event["feasible"]), key=lambda event: event["objective"])
        for name in ("fixed_cost", "expected_fuel_cost", "no_load_cost", "startup_cost", "shutdown_cost"):
            row[name] = best[name]
    return row


def make_figures(directory, traces, stops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    styles = [(0, 0, "#6b7280"), (1, 0, "#2563eb"), (0, 1, "#d97706"), (1, 1, "#059669")]
    for axis, grid, label in (("evaluations", np.arange(1, 41), "去重候选评估次数"),
                              ("seconds", np.arange(0, 301, 2), "运行时间（秒）")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for d in (0, 1):
            for column, case in enumerate(CASES):
                ax = axes[d, column]
                for q, l, color in styles:
                    means, stds = [], []
                    for budget in grid:
                        stats = describe([checkpoint(traces[(case, seed, f"q{q}_l{l}_d{d}")], axis, budget) for seed in SEEDS])
                        means.append(stats["mean"] if stats["n"] == len(SEEDS) else np.nan)
                        stds.append(stats["std"] or 0.0)
                    means, stds = np.asarray(means), np.asarray(stds)
                    ax.plot(grid, means, color=color, label=f"退火{'开启' if q else '关闭'}，大模型{'开启' if l else '关闭'}", linewidth=1.8)
                    ax.fill_between(grid, means - stds, means + stds, color=color, alpha=.10)
                    for seed in SEEDS:
                        stop = stops[(case, seed, f"q{q}_l{l}_d{d}")]
                        ax.scatter(stop[axis], stop['objective'], marker='x', s=18, color=color, alpha=.6)
                ax.set(title=f"{CASE_LABELS[case]}｜关机搜索{'开启' if d else '关闭'}", xlabel=label, ylabel="当前最优可行成本")
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                if d == 0 and column == 0:
                    ax.legend(fontsize=8)
        fig.suptitle("开悟退火＋大模型：均值与一个标准差；叉号表示各次运行停止点", fontsize=13)
        fig.savefig(figures / f"incumbent_by_{axis}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    grid = np.arange(1, 41)
    for ax, case in zip(axes, CASES):
        for strategy, label, color in (("q_only_raw", "原始退火（不修复）", "#6b7280"),
                                        ("q1_l0_d0", "退火＋修复", "#2563eb"),
                                        ("random_repair", "随机候选＋修复", "#9333ea")):
            stats = [describe([checkpoint(traces[(case, seed, strategy)], "evaluations", k) for seed in SEEDS]) for k in grid]
            values = [row['mean'] if row['n'] == len(SEEDS) else np.nan for row in stats]
            ax.plot(grid, values, label=label, color=color, linewidth=1.8)
            for seed in SEEDS:
                stop = stops[(case, seed, strategy)]
                ax.scatter(stop['evaluations'], stop['objective'], marker='x', s=18, color=color, alpha=.6)
        ax.set(title=CASE_LABELS[case], xlabel="去重候选评估次数", ylabel="最优可行成本均值")
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend(fontsize=8)
    fig.savefig(figures / "repair_controls.png", dpi=180)
    plt.close(fig)


def generate_report(output_dir):
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    references = {case: json.loads((directory / "references" / f"{case}.json").read_text(encoding="utf-8")) for case in CASES}
    metrics, traces, checkpoint_rows, stops = [], {}, [], {}
    for scheduled in manifest["schedule"]:
        run_id = scheduled["run_id"]
        record = json.loads((directory / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
        if not record["complete"] or record["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"incomplete or mismatched run: {run_id}")
        logs = {kind: read_jsonl(directory / "traces" / f"{run_id}.{kind}.jsonl")
                for kind in ("evaluations", "llm_calls", "sampling", "proposals")}
        traces[(record["case"], record["seed"], record["strategy"]["name"])] = logs["evaluations"]
        stops[(record["case"], record["seed"], record["strategy"]["name"])] = {
            'evaluations': record['diagnostics']['evaluation_count'],
            'seconds': record['result']['runtime'], 'objective': record['result']['objective']}
        metrics.append(run_metrics(record, logs["evaluations"], logs["llm_calls"], logs["sampling"],
                                   logs["proposals"], references[record["case"]]))
        for axis, budgets in (("evaluations", [10, 20, 40]), ("seconds", [30, 60, 120, 300])):
            for budget in budgets:
                checkpoint_rows.append({"case": record["case"], "seed": record["seed"],
                                        "strategy": record["strategy"]["name"], "axis": axis, "budget": budget,
                                        "objective": checkpoint(logs["evaluations"], axis, budget),
                                        "actual_evaluations": record["diagnostics"]["evaluation_count"],
                                        "actual_runtime": record["result"]["runtime"]})
    summaries = []
    summary_metrics = ("objective", "runtime", "reference_gap_pct", "relative_error_pct", "evaluations",
                       "fixed_cost", "expected_fuel_cost", "llm_evaluated", "q_raw_feasible_rate", "q_unique_u_per_round")
    for case in CASES:
        for strategy in STRATEGIES:
            members = [row for row in metrics if row["case"] == case and row["strategy"] == strategy["name"]]
            summary = {"case": case, "strategy": strategy["name"], "runs": len(members),
                       "feasible_runs": sum(row["feasible"] for row in members),
                       "llm_failed_calls": sum(row["llm_failed_calls"] for row in members)}
            for metric in summary_metrics:
                summary.update({f"{metric}_{key}": value for key, value in describe([row.get(metric) for row in members]).items()})
            summaries.append(summary)
    effects = []
    for case in CASES:
        for d in (0, 1):
            for axis, budgets in (("evaluations", [10, 20, 40]), ("seconds", [30, 60, 120, 300])):
                for budget in budgets:
                    costs = [[checkpoint(traces[(case, seed, f"q{q}_l{l}_d{d}")], axis, budget)
                              for q, l in ((0, 0), (1, 0), (0, 1), (1, 1))] for seed in SEEDS]
                    costs = [[np.nan if value is None else value for value in row] for row in costs]
                    effects.extend({"case": case, "d": d, "axis": axis, "budget": budget, **row} for row in paired_effects(costs))
    controls = []
    contrasts = [("repair_gain", "q_only_raw", "q1_l0_d0"),
                 ("q_over_random", "random_repair", "q1_l0_d0")]
    contrasts += [(f"d_gain_q{q}_l{l}", f"q{q}_l{l}_d0", f"q{q}_l{l}_d1")
                  for q in (0, 1) for l in (0, 1)]
    for case in CASES:
        for axis, budgets in (("evaluations", [10, 20, 40]), ("seconds", [30, 60, 120, 300])):
            for budget in budgets:
                for name, before, after in contrasts:
                    values = [[checkpoint(traces[(case, seed, strategy)], axis, budget)
                               for seed in SEEDS] for strategy in (before, after)]
                    controls.append({"case": case, "axis": axis, "budget": budget, "effect": name,
                                     **paired_difference(*values)})
    total_calls = sum(row["llm_calls"] for row in metrics)
    requests = sum(row["llm_requests"] for row in metrics)
    ledger_path = directory / "execution-ledger.jsonl"
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    charged_calls = sum(event.get("logical", 0) for event in ledger) if ledger else total_calls
    charged_requests = sum(event.get("requests", 0) for event in ledger) if ledger else requests
    if len(metrics) != 150 or max(total_calls, charged_calls) > 600 or max(requests, charged_requests) > 1200:
        raise AssertionError("suite size or API budget mismatch")
    write_csv(directory / "runs.csv", metrics)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "checkpoints.csv", checkpoint_rows)
    write_csv(directory / "effects.csv", effects)
    write_csv(directory / "controls.csv", controls)
    write_json(directory / "analysis.json", {"generated_at": now(), "fingerprint": manifest["fingerprint"],
               "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "summaries": summaries, "effects": effects, "controls": controls, "references": references,
               "charged_logical_calls": charged_calls, "charged_requests": charged_requests})
    make_figures(directory, traces, stops)
    fmt = lambda value, digits=2: "—" if value is None else f"{value:,.{digits}f}"
    lines = ["# Kaiwu 退火＋LLM 主问题候选贡献实验", "",
             f"生成时间：{now()}。实验指纹：\u0060{manifest['fingerprint']}\u0060。", "",
             "本实验评估 Kaiwu 本地模拟退火（量子启发式）、LLM、确定性修复及贪心关机搜索的作用。"
             "Q、L、D 分别表示前三者中的退火、LLM、关机搜索；主实验统一保留修复和确定性初始候选。", "",
             f"完成 {len(metrics)} 次研究运行及 3 次独立完整模型求解；返回并通过物理检查的方案 {sum(row['feasible'] for row in metrics)}/150。"
             f"研究运行累计 {sum(row['runtime'] for row in metrics)/3600:.2f} 小时，不含独立参考求解。", "",
             f"LLM 使用 {manifest['settings']['llm']['model']}：{total_calls} 次逻辑调用、{requests} 次请求，"
             f"失败调用 {sum(row['llm_failed_calls'] for row in metrics)} 次。"
             f"其中输出截断 {sum(row['llm_truncated_calls'] for row in metrics)} 次；"
             "方案可行率与 LLM 调用成功率分开计算，失败调用不产生已接受候选，其耗时和用量仍计入。"
             f"已报告输入/输出 token 共 {sum(row['prompt_tokens_known'] for row in metrics):,}/"
             f"{sum(row['completion_tokens_known'] for row in metrics):,}；"
             f"缺失用量的调用 {sum(row['usage_unknown_calls'] for row in metrics)} 次。未估算货币费用。", "",
             "## 设计与解释", "",
             "三个工况为原始负荷、全部负荷降低 10%、全部负荷循环后移 2 小时；均为 10×24、三个场景和 5% 备用。"
             "种子 7、17、27、37、47；每次最多 10 轮及 40 个唯一候选评估。运行同机串行，Gurobi 单线程。"
             "未耗尽预算时保留实际停止点，并沿用已获得的 incumbent；不补造评估。", "",
             "LLM 远端采样不承诺按本地种子确定性复现；重复编号用于实验分块和配对。确定性组的重复仅用于计时。"
             "失败调用保留在端到端结果中，单独报告，不以零值替代缺失成本。", "",
             "固定候选预算由所有来源共享，评估顺序保持现有实现。开启一种来源会改变候选、incumbent、"
             "后续搜索和剩余额度；消融效应衡量这一完整策略变化，不等同于单个建议的独立贡献。", "",
             "完整模型在全部研究运行结束后求解，目标容差 1e-4、时限 300 秒。QUBO 能量和研究候选均不提供认证下界。", "",
             "## 独立参考", "", "| 工况 | 可行目标 | ObjBound | 状态 | MIP gap | 秒 |", "|---|---:|---:|---|---:|---:|"]
    for case, ref in references.items():
        lines.append(f"| {case} | {fmt(ref['objective'])} | {fmt(ref['lower_bound'])} | {ref['status']} | {fmt(ref['mip_gap'],6)} | {fmt(ref['runtime'])} |")
    lines += ["", "## 各组结果", "", "目标和耗时为五次重复的均值；± 为样本标准差。差距使用独立 ObjBound。"]
    for case in CASES:
        lines += ["", f"### {case}", "", "| 组 | 目标均值 ± 标准差 | 差距 % | 时间均值秒 | 唯一评估 | LLM 新评估 |", "|---|---:|---:|---:|---:|---:|"]
        for row in [item for item in summaries if item["case"] == case]:
            lines.append(f"| {row['strategy']} | {fmt(row['objective_mean'])} ± {fmt(row['objective_std'])} | {fmt(row['reference_gap_pct_mean'])} | {fmt(row['runtime_mean'])} | {fmt(row['evaluations_mean'],1)} | {fmt(row['llm_evaluated_mean'],1)} |")
    lines += ["", "## 退火、LLM 增益与协同", "",
              "下表固定 40 次评估预算，分别在 D 关闭、开启时计算。Q 增益为 C01−C11，L 增益为 C10−C11；"
              "交互 I=C10+C01−C00−C11。正值表示降低成本；I>0 表示超加性收益。括号为配对 bootstrap 95% 区间，重采样 10,000 次。", "",
              "| 工况 | D | Q 增益（已有 L） | L 增益（已有 Q） | 交互 I |", "|---|---:|---:|---:|---:|"]
    for case in CASES:
        for d in (0, 1):
            subset = {row["effect"]: row for row in effects if row["case"] == case and row["d"] == d and row["axis"] == "evaluations" and row["budget"] == 40}
            cells = [f"{fmt(subset[key]['mean'])} [{fmt(subset[key]['ci_low'])}, {fmt(subset[key]['ci_high'])}]" for key in ("q_gain_with_l", "l_gain_with_q", "interaction")]
            lines.append(f"| {case} | {d} | " + " | ".join(cells) + " |")
    lines += ["", "## 修复、随机对照与关机搜索的贡献", "",
              "以下为 40 次评估预算下的配对成本降低量及 bootstrap 95% 区间。正值表示后者更好。"
              "修复增益比较 Q+R 与原始 Q；退火相对随机比较 Q+R 与随机+R；D 增益在相同 Q/L 配置内比较。"
              "时间预算及其他评估预算的全部比较见 controls.csv。", "",
              "| 工况 | 比较 | 平均降低成本 | 95% 区间 |", "|---|---|---:|---:|"]
    for row in controls:
        if row["axis"] == "evaluations" and row["budget"] == 40:
            lines.append(f"| {row['case']} | {row['effect']} | {fmt(row['mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |")
    lines += ["", "## 数据支持的结论", ""]
    for case in CASES:
        best = min((row for row in summaries if row['case'] == case and row['objective_mean'] is not None),
                   key=lambda row: row['objective_mean'])
        lines.append(f"{case}：最终平均目标值最低的组为 {best['strategy']}（{fmt(best['objective_mean'])}）。"
                     "这是本次样本内的排序，不等同于统计显著优于其余组。")
        for name, label in [('repair_gain', '修复相对原始退火'), ('q_over_random', '退火相对随机候选（均修复）')]:
            row = next(row for row in controls if row['case'] == case and row['effect'] == name
                       and row['axis'] == 'evaluations' and row['budget'] == 40)
            conclusion = ('缺少足够配对样本' if row['ci_low'] is None else
                          '本次配对区间支持成本降低' if row['ci_low'] > 0 else
                          '本次配对区间支持成本升高' if row['ci_high'] < 0 else
                          '区间包含零，尚不能确认优劣')
            lines.append(f"{label}：平均成本降低 {fmt(row['mean'])}，{conclusion}。")
        for d in (0, 1):
            selected = {row['effect']: row for row in effects if row['case'] == case and row['d'] == d
                        and row['axis'] == 'evaluations' and row['budget'] == 40}
            observations = []
            for key, label in [('q_gain_with_l', '已有 L 时加入 Q'), ('l_gain_with_q', '已有 Q 时加入 L'),
                               ('interaction', 'Q/L 交互')]:
                row = selected[key]
                evidence = ('缺少足够配对样本' if row['ci_low'] is None else
                            '区间完全为正' if row['ci_low'] > 0 else
                            '区间完全为负' if row['ci_high'] < 0 else '区间包含零，方向证据不足')
                observations.append(f"{label}均值 {fmt(row['mean'])}，{evidence}")
            lines.append(f"D={d}、40 次评估预算：" + '；'.join(observations) + '。')
        lines.append('')
    quantum = [row for row in metrics if row["q_raw_candidates"]]
    raw_count = sum(row["q_raw_candidates"] for row in quantum)
    feasible_count = sum(row["q_raw_feasible"] for row in quantum)
    lines += ["", "## 候选与修复诊断", "",
              f"退火共输出 {raw_count} 个原始候选，离散可行 {feasible_count} 个（{100*feasible_count/max(1,raw_count):.2f}%）。"
              f"LLM 共返回 {sum(row['llm_raw_candidates'] for row in metrics)} 个建议，修复和局部去重后保留 "
              f"{sum(row['llm_valid_candidates'] for row in metrics)} 个，最终进入新场景评估 "
              f"{sum(row['llm_evaluated'] for row in metrics)} 个。", "",
              "来源标签描述实际评估顺序；跨来源重复方案不重复计功，算法贡献以对照及配对效应为依据。"
              "原始/修复后方案、汉明距离、缓存命中、阶段耗时和截断等信息见逐运行记录与 traces。", "",
              "## 曲线", "", "实线为五次重复均值，阴影为一个样本标准差；叉号为各运行的实际停止点及最终目标值。"
              "曲线从全部五次重复均已找到可行解的预算开始显示，避免早期缺失样本改变均值的样本组成。"
              "时间曲线仅使用当时已经完成的评估。", "",
              "![候选评估预算](figures/incumbent_by_evaluations.png)", "",
              "![实际时间预算](figures/incumbent_by_seconds.png)", "",
              "![修复与随机对照](figures/repair_controls.png)", "",
              "## 证据边界与复现", "",
              "这是三个相关工况、五次重复下的探索性证据。区间未包含模型选择、远端服务漂移或实例总体的不确定性，"
              "多个比较未进行多重检验校正，不能据此宣称量子硬件优势或跨规模的一般结论。"
              "各时间和候选预算下的效应完整保存在 effects.csv。", "",
              "manifest.json、source、working-tree.patch 和 inputs 固定本次代码、未提交修复、参数与数据。"
              "runs 保存结果；traces 保存候选、退火诊断及去除鉴权信息的 LLM 请求/响应；CSV 为汇总分析。"
              "进程环境提供 LLM_API_KEY，使用 README 中的 attribution 命令及 --resume 恢复相同配置的已完成运行。", ""]
    if ledger:
        lines += ["## 中断与总调用核算", "",
                  "首个实验块的第二组曾因交互进程中断而重跑，未完成的数据不纳入算法统计。"
                  "中断前观察到 5 次逻辑调用和 5 次请求，其中第 5 次结果未知；其响应及 token 用量未完整落盘。"
                  "为覆盖未观察到的消耗，账本为该次中断保守预留 10 次逻辑调用、20 次请求。", "",
                  f"上述完成样本的调用量之外，计入中断预留后的预算记账为 {charged_calls}/600 次逻辑调用、"
                  f"{charged_requests}/1200 次请求。预留量并非实际消耗估计；已知 token 总量不包含遗失用量。"
                  "后续调用及脱敏诊断逐次持久化到 execution-ledger.jsonl。", "",
                  f"除此预留外，账本记录但未纳入完成样本的额外逻辑调用为 {charged_calls-total_calls-10} 次、"
                  f"请求为 {charged_requests-requests-20} 次（包括后续中断重跑开销）。"
                  "恢复记录见 recovery.json；这些未完成样本不计入算法效果或已完成运行耗时。", ""]
    # Localize presentation only; original identifiers remain in the research records.
    labels = {**CASE_LABELS, **STRATEGY_LABELS,
              "repair_gain": "修复增益", "q_over_random": "退火相对随机候选的增益"}
    for q in (0, 1):
        for l in (0, 1):
            labels[f"d_gain_q{q}_l{l}"] = f"关机搜索增益（退火{'开启' if q else '关闭'}，大模型{'开启' if l else '关闭'}）"
    labels.update({
        "deepseek-v4.1-flash": "深度求索第4.1版闪速模型",
        "Kaiwu": "开悟", "LLM": "大模型", "Gurobi": "数学规划求解器",
        "ObjBound": "求解器下界", "MIP gap": "整数规划相对间隙",
        "OPTIMAL": "已达最优容差", "TIME_LIMIT": "达到时间上限",
        " incumbent": "当前最优可行解", "、incumbent、": "、当前最优可行解、", "bootstrap": "自助重采样",
        "token": "词元", "QUBO": "二次无约束二进制模型",
        "Q+R": "退火＋修复", "原始 Q": "原始退火", "随机+R": "随机＋修复",
        "Q/L": "退火与大模型", "D=0": "关机搜索关闭", "D=1": "关机搜索开启",
        "D 关闭、开启": "关机搜索关闭、开启", "D 增益": "关机搜索增益",
        "Q 增益": "退火增益", "L 增益": "大模型增益",
        "已有 L": "已有大模型", "已有 Q": "已有退火", "加入 Q": "加入退火", "加入 L": "加入大模型",
        "Q、L、D 分别表示前三者中的退火、LLM、关机搜索；": "",
        "| D |": "| 关机搜索（０为关闭，１为开启） |",
        "LLM_API_KEY": "大模型接口凭证环境变量",
    })
    # Replace longer identifiers first so abbreviations cannot corrupt file names.
    report = "\n".join(lines)
    for english, chinese in sorted(labels.items(), key=lambda item: -len(item[0])):
        report = report.replace(english, chinese)
    links = {
        "controls.csv": "修复与关机搜索对照表", "effects.csv": "增益与交互统计表",
        "manifest.json": "实验参数清单", "source": "代码快照", "working-tree.patch": "工作区修改快照",
        "inputs": "输入数据", "runs": "逐运行结果", "traces": "候选与调用日志",
        "execution-ledger.jsonl": "调用核算账本", "recovery.json": "中断恢复记录",
    }
    for filename, title in links.items():
        report = report.replace(filename, f"[{title}]({filename})")
    report = report.replace("CSV 为汇总分析", "汇总表保存统计分析")
    report = report.replace("使用 README 中的 attribution 命令及 --resume 恢复相同配置的已完成运行",
                            "按[项目使用说明](../../README.md)中的实验与恢复命令继续相同配置的运行")
    (directory / "report.md").write_text(report, encoding="utf-8")
    return {"runs": len(metrics), "llm_calls": total_calls, "llm_requests": requests}
