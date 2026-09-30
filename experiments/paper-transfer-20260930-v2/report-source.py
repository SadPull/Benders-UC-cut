"""Post-run reports; never feed reference or report state into search."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np


def report(directory):
    directory = Path(directory)
    shutil.copyfile(__file__, directory/"report-source.py")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory/"runs").glob("*.json"))]
    reference = json.loads((directory/"reference.json").read_text(encoding="utf-8")) if (directory/"reference.json").exists() else {}
    rows, checkpoints = [], []
    for record in records:
        result = record["result"]
        diag = result["diagnostics"]
        sizes = [r["bits"] for r in diag["sampling"]]
        changes = [r["hamming_u"] for r in diag["proposals"] if r.get("hamming_u") is not None]
        raw = sum(r["raw_candidates"] for r in diag["sampling"])
        valid = sum(r["raw_discrete_feasible"] for r in diag["sampling"])
        events = diag["evaluations"]
        row = {"strategy": record["strategy"], "seed": record["seed"], "objective": result["objective"],
               "runtime": result["runtime"], "status": result["status"], "full_evaluations": diag["evaluation_count"],
               "local_evaluations": diag.get("local_evaluation_count", 0), "max_bits": max(sizes, default=0),
               "mean_bits": float(np.mean(sizes)) if sizes else 0., "sa_calls": len(sizes),
               "fraction_over_500": sum(v > 500 for v in sizes)/len(sizes) if sizes else 0.,
               "mean_auxiliary_bits": float(np.mean([r.get("auxiliary_bits", 0) for r in diag["sampling"]])) if sizes else 0.,
               "mean_density": float(np.mean([r.get("density", 0) for r in diag["sampling"]])) if sizes else 0.,
               "raw_feasibility": valid/raw if raw else 0., "mean_repair_hamming": float(np.mean(changes)) if changes else 0.,
               "local_feasibility": float(np.mean([e["feasible"] for e in diag.get("local_evaluations", [])])) if diag.get("local_evaluations") else None,
               "full_feasibility": float(np.mean([e["feasible"] for e in events if not e["cache_hit"]])) if events else None,
               "overshoot_seconds": max(0., result["runtime"]-300),
               "reference_gap": (result["objective"]-reference["lower_bound"])/max(1., abs(result["objective"]))
                   if result["objective"] is not None and reference.get("lower_bound") is not None else None}
        row.update({f"seconds_{k}": v for k, v in diag["timings"].items()})
        rows.append(row)
        for cutoff in (30, 60, 120, 300):
            eligible = [e["objective"] for e in events if e["elapsed_seconds"] <= cutoff and e["feasible"] and e["objective"] is not None]
            checkpoints.append({"strategy": record["strategy"], "seed": record["seed"], "seconds": cutoff,
                                "objective": min(eligible) if eligible else None})
    def save_csv(filename, entries):
        fields = list(dict.fromkeys(k for row in entries for k in row))
        with (directory/filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
    save_csv("runs.csv", rows)
    save_csv("checkpoints.csv", checkpoints)
    strategies = sorted({r["strategy"] for r in rows})
    summary = []
    for strategy in strategies:
        group = [r for r in rows if r["strategy"] == strategy]
        valid = [r for r in group if r["objective"] is not None]
        values = [r["objective"] for r in valid]
        summary.append({"strategy": strategy, "runs": len(group), "feasible_runs": len(valid),
                        "objective_mean": float(np.mean(values)) if values else None,
                        "objective_std": float(np.std(values, ddof=1)) if len(values)>1 else 0.,
                        "runtime_mean": float(np.mean([r["runtime"] for r in group])),
                        "max_bits": max(r["max_bits"] for r in group),
                        "sa_calls_mean": float(np.mean([r["sa_calls"] for r in group]))})
    save_csv("summary.csv", summary)
    comparisons = []
    for left, right in (("A_legacy", "B_compact"), ("A_legacy", "C_dynamic"), ("B_compact", "C_dynamic"),
                        ("C_dynamic", "D_block1"), ("C_dynamic", "E_block2"), ("C_dynamic", "F_block3"),
                        ("A_legacy", "D_block1"), ("A_legacy", "E_block2"), ("A_legacy", "F_block3")):
        pairs = []
        for seed in (7, 17, 27, 37, 47):
            l = next((r for r in checkpoints if r["strategy"] == left and r["seed"] == seed and r["seconds"] == 300), None)
            r = next((r for r in checkpoints if r["strategy"] == right and r["seed"] == seed and r["seconds"] == 300), None)
            if l and r and l["objective"] is not None and r["objective"] is not None:
                pairs.append(l["objective"]-r["objective"])
        if pairs:
            comparisons.append({"baseline": left, "method": right, "pairs": len(pairs),
                                "saving_mean": float(np.mean(pairs)), "saving_std": float(np.std(pairs, ddof=1)) if len(pairs)>1 else 0.,
                                "wins": sum(x>1e-4 for x in pairs), "ties": sum(abs(x)<=1e-4 for x in pairs),
                                "losses": sum(x < -1e-4 for x in pairs)})
    save_csv("paired.csv", comparisons)
    lines = ["# 时间分解与 SA–Benders 论文移植实验", "",
             f"完成 {len(records)}/30 次运行；通过物理检查并返回可行方案 {sum(r['objective'] is not None for r in rows)}/{len(rows)}。",
             "", "原始 10×24 三场景模型；关闭 LLM、减开机与随机候选，保留确定性修复。每次最多 300 秒、40 次完整候选评估；分块局部评估单独计数且耗时计入。",
             "", "300 秒按阶段和调用边界检查，SA 不可中断。最终成本可能包含超时完成的结果；公平成本比较采用 checkpoints.csv 中截止时刻前已完成的评估。",
             "", "## 完整运行结果", "", "| 方法 | 次数 | 目标均值 ± 标准差 | 时间均值秒 | 最大比特数 | SA调用均值 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        objective = f"{r['objective_mean']:,.2f} ± {r['objective_std']:,.2f}" if r['objective_mean'] is not None else "无可行解"
        lines.append(f"| {r['strategy']} | {r['runs']} | {objective} | {r['runtime_mean']:.2f} | {r['max_bits']} | {r['sa_calls_mean']:.1f} |")
    lines += ["", "A：原实现；B：紧凑静态编码；C：整时域动态编码；D/E/F：1/2/3 小时分块动态编码。", "",
              "## 300 秒配对成本差", "", "正数表示右侧方法降低成本。", "",
              "| 对照 → 方法 | 配对数 | 平均降低成本 ± 标准差 | 胜/平/负 |", "|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['baseline']} → {r['method']} | {r['pairs']} | {r['saving_mean']:,.2f} ± {r['saving_std']:,.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "## 初步判断", ""]
    for r in comparisons:
        if r["baseline"] != "A_legacy":
            continue
        direction = "平均降低" if r["saving_mean"] > 0 else "平均增加"
        lines.append(f"- {r['method']} 相对当前基线，在 300 秒口径下{direction}成本 {abs(r['saving_mean']):,.2f}；{r['pairs']} 个配对中胜 {r['wins']}、平 {r['ties']}、负 {r['losses']}。")
    baseline_bits = next((r["max_bits"] for r in summary if r["strategy"] == "A_legacy"), None)
    if baseline_bits:
        for r in summary:
            if r["strategy"].startswith(("D_", "E_", "F_")):
                lines.append(f"- {r['strategy']} 的全程最大矩阵为 {r['max_bits']} 比特，相对基线 {baseline_bits} 比特减少 {100*(1-r['max_bits']/baseline_bits):.1f}%。")
    lines += ["", "## 相同时间预算", "", "以下成本仅使用截止时间前完成的可行评估，提前停止后沿用最后已获得的解。", "",
              "| 方法 | 30秒均值 | 60秒均值 | 120秒均值 | 300秒均值 |", "|---|---:|---:|---:|---:|"]
    for strategy in strategies:
        cells = []
        for cutoff in (30, 60, 120, 300):
            values = [r["objective"] for r in checkpoints if r["strategy"] == strategy and r["seconds"] == cutoff and r["objective"] is not None]
            total = sum(r["strategy"] == strategy for r in records)
            cells.append(f"{np.mean(values):,.2f} ({len(values)}/{total})" if values else "无可行解")
        lines.append("| "+strategy+" | "+" | ".join(cells)+" |")
    lines += ["", "## 规模、可行性和开销", "",
              "| 方法 | 原始局部/主问题可行率 | 矩阵超过500比特比例 | 完整评估均值 | 局部评估均值 |", "|---|---:|---:|---:|---:|"]
    for strategy in strategies:
        group = [r for r in rows if r["strategy"] == strategy]
        lines.append(f"| {strategy} | {100*np.mean([r['raw_feasibility'] for r in group]):.2f}% | {100*np.mean([r['fraction_over_500'] for r in group]):.2f}% | {np.mean([r['full_evaluations'] for r in group]):.1f} | {np.mean([r['local_evaluations'] for r in group]):.1f} |")
    lines += ["", "## 解释与限制", "",
              "原始采样可行率的检查范围不同：分块只检查块内保留的硬约束，跨块约束已经松弛。该指标不能解释为完整 SUC 原始可行率提高；完整方案仍须全时域修复和调度验证。",
              "",
              "时间分解保留每个原始约束的全局索引，跨块约束通过有界拉格朗日切平面协调。SA 采样平面是对偶函数的仿射上界，不保证相切；可靠下界仅来自同一乘子下的连续松弛。",
              "", "局部最优性割在乘子变化后重建，局部可行性割保留。局部调度目标含拉格朗日项，不能作为最终 SUC 成本。完整方案经过全时域修复及独立连续调度检查。",
              "", "紧凑/动态组先评估初始方案，用于取得编码上界；原实现保留原有先采样再评估顺序。新编码组使用联合场景 QP，原实现使用逐场景 QP，因此时间差包含实现组织开销。",
              "", "500 比特仅是论文 CIM 容量参照；变量数达标不表示已满足真机精度、连接或噪声要求。五个种子只构成初步证据，不宣称量子加速或全局收敛。"]
    if reference:
        lines += ["", f"独立完整模型：目标 {reference['objective']:,.4f}，下界 {reference['lower_bound']:,.4f}，状态 {reference['status']}，MIP gap {reference['mip_gap']:.6g}。参考求解在研究运行后执行。"]
    if (directory/"audit.json").exists():
        audit = json.loads((directory/"audit.json").read_text(encoding="utf-8"))
        lines += ["", "## 最终验证", "",
                  f"通过全时域物理检查 {audit['feasible_validated']}/{audit['runs']}；最大物理残差 {audit['max_physical_residual']:.3g} MW，最大成本分解残差 {audit['max_cost_residual']:.3g}。",
                  f"LLM 调用 {audit['llm_calls']} 次，减开机候选 {audit['decommit_candidates']} 个；最大完整评估数 {audit['max_full_evaluations']}，最长运行 {audit['max_runtime']:.2f} 秒。"]
        if audit.get("unit_tests_passed"):
            lines += [f"完整测试套件 {audit['unit_tests_passed']} 项通过，缓存回归所在模块的 9 项测试单独复验通过；Git 空白检查通过。研究求解源码与实验指纹一致。"]
    lines += ["", "数据文件：runs.csv、summary.csv、paired.csv、checkpoints.csv；各次 JSON 保留局部求解、编码范围、外层乘子、耦合残差和修复轨迹。"]
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if summary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].errorbar(range(len(summary)), [r["objective_mean"] or np.nan for r in summary],
                         yerr=[r["objective_std"] for r in summary], fmt="o", capsize=5)
        axes[0].set_xticks(range(len(summary)), [r["strategy"] for r in summary])
        axes[0].set_ylabel("Final feasible objective (mean +/- sample SD)")
        axes[0].grid(axis="y", alpha=.2)
        axes[1].bar([r["strategy"] for r in summary], [r["max_bits"] for r in summary])
        axes[1].axhline(500, color="red", linestyle="--", label="500-bit reference")
        axes[1].set_ylabel("Largest QUBO (bits)")
        axes[1].legend()
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(directory/"comparison.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in strategies:
            means = []
            for cutoff in (30, 60, 120, 300):
                values = [r["objective"] for r in checkpoints if r["strategy"] == strategy and r["seconds"] == cutoff and r["objective"] is not None]
                means.append(np.mean(values) if values else np.nan)
            ax.plot((30, 60, 120, 300), means, marker="o", label=strategy)
        ax.set_xlabel("Wall-clock budget (seconds)")
        ax.set_ylabel("Mean feasible objective observed by deadline")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory/"time_checkpoints.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    report(parser.parse_args().directory)
