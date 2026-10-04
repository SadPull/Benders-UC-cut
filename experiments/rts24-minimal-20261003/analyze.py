"""Post-run analysis only. Reads validated summaries; never generates candidates."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
summary_path = ROOT / "final-summary.json"
if not summary_path.exists():
    summary_path = ROOT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = summary["runs"]
reference = summary["reference_cost"]
colors = {7: "#2166ac", 17: "#b2182b", 27: "#238b45"}
names = {"A": "Fixed-seed SA", "B": "Changing-seed SA", "C": "LLM only",
         "D": "SA + LLM", "E": "SA + rule heuristic"}

def run_label(run):
    label = f"Seed {run['seed']}"
    if run.get("replacement"):
        label += " (replacement)"
    if run.get("llm_transport_status") == "no_response":
        label += " (no LLM response)"
    return label

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True)
for group, ax in zip(names, axes.flat):
    for run in rows:
        if run["group"] != group or not run["curve"]:
            continue
        points = run["curve"]
        times = [p["seconds"] for p in points] + [1800]
        gaps = [100*(p["objective"]/reference-1) for p in points]
        ax.step(times, gaps+[gaps[-1]], where="post", color=colors[run["seed"]], label=run_label(run))
    ax.axhline(1, color="#777777", linestyle="--", linewidth=.8)
    ax.axhline(.5, color="#aaaaaa", linestyle=":", linewidth=.8)
    ax.set_title(f"{group}: {names[group]}")
    ax.set_xlim(0, 1800)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 300, 600, 900, 1200, 1800])
    ax.grid(alpha=.2)
    ax.set_xlabel("Search time (s)")
    ax.set_ylabel("Difference from reference (%)")
    if ax.lines:
        ax.legend(fontsize=8)
axes.flat[-1].axis("off")
axes.flat[-1].text(0, .95, "RTS-24 / three scenarios\n\nThree local seeds per method\n1800 s search budget\nC stops after 3 LLM calls\n\nDashed line: 1%\nDotted line: 0.5%\n\nExploratory evidence only;\nno significance claim.", va="top", transform=axes.flat[-1].transAxes)
fig.savefig(ROOT / "quality-curves.png", dpi=180)
fig.savefig(ROOT / "quality-curves.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, constrained_layout=True)
for group, ax in zip(("C", "D"), axes):
    for run in rows:
        if run["group"] != group or not run["curve"]:
            continue
        points = run["curve"]
        times = [p["seconds"] for p in points] + [1800]
        gaps = [100*(p["objective"]/reference-1) for p in points]
        ax.step(times, gaps+[gaps[-1]], where="post", color=colors[run["seed"]], label=run_label(run))
    ax.axhline(1, color="#777777", linestyle="--", linewidth=.8)
    ax.axhline(.5, color="#aaaaaa", linestyle=":", linewidth=.8)
    ax.set(title=f"{group}: {names[group]}", xlim=(0, 1800), ylim=(0, 2),
           xlabel="Search time (s)", ylabel="Difference from reference (%)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("Near-reference view (0–2%); higher values are outside the plot")
fig.savefig(ROOT / "quality-detail.png", dpi=180)
fig.savefig(ROOT / "quality-detail.svg")
plt.close(fig)

lines = ["# RTS-24 最小必要性实验：最终结果", "", f"最终分析集数值验收：{summary['passed']}。", "",
         "**当前证据不足以证明 SA 与 LLM 的双向协同，也不足以支持 SA 的必要性或相对 Gurobi 的加速主张。**",
         "已完成原批次及两次授权补跑，共 17 次搜索尝试和一次独立参考；最终保留 15 个预定组别/种子结果。运行结束不等于有效模型重复已补齐。",
         f"六次入选 C/D 运行中，{summary.get('selected_llm_runs_with_response', 0)} 次至少接收了一次模型响应，{summary.get('selected_llm_runs_all_three_received', 0)} 次接收了全部三次响应；C/D 两侧均接收到模型响应的配对种子为 {summary.get('paired_seeds_with_llm_response', [])}。",
         "种子 27 的 D 最终参考差为 0.1014%，低于 C 的 0.8568%，但只有这一对同时接收到模型响应，且 C 第三次调用超时、D 三次均接收，不能据此识别 SA 本身的贡献。",
         "D／27 在首次 LLM 改善后又执行 7 次 SA、完成 2 次完整评估，未进一步降低最佳成本。D／17 补跑没有执行后续 SA；D／7 没有接收 LLM 响应。这些记录不能替代充分的 SA 后续贡献实验。",
         "收到响应的 C/D 运行达到 0.1014%–0.8568% 参考差；A/B 为 28.5462%–34.0872%，E 为 21.2999%–28.5622%。它们支持在本算例和当前实现中继续研究经验证的 LLM 候选，但不证明一般传统启发式无法替代 LLM。",
         "全部尝试共启动 18 次模型请求，低于授权总上限 24 次；未用完的额度不表示已批准额外运行。", "",
         "原 D／17 的控制进程和工作进程意外退出，第三次模型输出未被调度验证；原始部分结果完整保留。",
         "用户授权在原批次结束后，以同一冻结源码和配置补跑 D／17，模型调用总上限由 18 调整为 21。",
         "C／17 首次请求被认领后，截止前没有模型输出或回执，原因未知；原始记录保留。用户另行批准补跑 C／17，总调用上限提高到 24。",
         "下表使用 15 次有正常结束记录的运行；D／17、C／17 使用明确标记的替代运行。全部 17 次尝试及失败记录见 final-summary.json。",
         "数值审计通过仅说明记录、约束、成本和预算核对通过，不表示全部模型调用成功；C／27 的第三次调用仍因截止未计分。",
         "D／7 的首次输出未在截止前完成回执交接，整次运行没有接收任何 LLM 响应；其 32.7028% 参考差反映交接失败后的端到端结果，不能解释为模型候选质量。该运行保留在预定分析集中，并明确标记。", "",
         "D／17 补跑的第三次输出同样没有在截止前交回；C／17 补跑的首次请求在认领前等待约 1175 秒，随后输出虽已写入，截止前仍没有完成通知或回执，最终只保留初始解。两次补跑仍受到中断、模型会话及交接延迟影响，不能视为无故障的算法测量。", "",
         "| 方法 | 结束记录数 | 参考差中位数 | 最小–最大参考差 | ≤1% 达标 | ≤0.5% 达标 |", "|---|---:|---:|---|---:|---:|"]
for group in names:
    runs = [r for r in rows if r["group"] == group and r["reference_difference"] is not None]
    if not runs:
        continue
    gaps = [100*r["reference_difference"] for r in runs]
    lines.append(f"| {group} {names[group]} | {len(runs)} | {statistics.median(gaps):.4f}% | {min(gaps):.4f}%–{max(gaps):.4f}% | {sum(r['threshold_seconds']['0.01'] is not None for r in runs)}/{len(runs)} | {sum(r['threshold_seconds']['0.005'] is not None for r in runs)}/{len(runs)} |")
lines += ["", "## 逐次结果", "", "LLM 请求次数包含未成功返回的请求；成功接收不保证所有候选都赶上完整调度验证。", "",
          "| 方法/种子 | 运行目录 | 成本 | 参考差 | ≤1% 秒 | ≤0.5% 秒 | SA 次数 | LLM 请求/接收 | LLM 交接状态 | LLM 后 SA 降本 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
transport_names = {"not_applicable": "不适用", "no_response": "无响应", "partial": "部分接收", "all_three_received": "三次接收"}
for r in sorted(rows, key=lambda r: (r["group"], r["seed"])):
    thresholds = ["—" if t is None else f"{t:.2f}" for t in r["threshold_seconds"].values()]
    lines.append(f"| {r['group']}/{r['seed']} | {r.get('run_directory', '')} | {r['objective']:.3f} | "
                 f"{100*r['reference_difference']:.4f}% | {' | '.join(thresholds)} | {r['sa_calls']} | "
                 f"{r['llm_started']}/{r.get('llm_responses_consumed', 0)} | "
                 f"{transport_names.get(r.get('llm_transport_status'), '待最终审计')} | {r['sa_improvement_after_llm']:.3f} |")
lines += ["", "## 配对比较", "", "下表为 D 成本减去对照成本，负数表示 D 更好；配对仅按本地种子，不代表模型随机性受控。", "",
          "| 种子 | D − C | D − B | D − E | LLM 改进后 SA 调用/完整评估 | D 首次 LLM 改进后 SA 新增降本 |", "|---|---:|---:|---:|---:|---:|"]
by_pair = {(r["group"], r["seed"]): r for r in rows}
for seed in (7, 17, 27):
    d = by_pair.get(("D", seed))
    if not d or d["objective"] is None:
        continue
    differences = []
    for group in ("C", "B", "E"):
        comparator = by_pair.get((group, seed))
        differences.append(f"{d['objective']-comparator['objective']:.3f}" if comparator and comparator["objective"] is not None else "—")
    post = d.get("post_llm_sa", {})
    lines.append(f"| {seed} | " + " | ".join(differences) +
                 f" | {post.get('calls', 0)}/{post.get('complete_evaluations', 0)} | {d['sa_improvement_after_llm']:.3f} |")
lines += ["", "## 固定时间检查点", "", "| 方法/种子 | 300 秒 | 600 秒 | 900 秒 | 1800 秒 |", "|---|---:|---:|---:|---:|"]
for r in sorted(rows, key=lambda r: (r["group"], r["seed"])):
    values = ["—" if r["checkpoints"][str(t)] is None else f"{r['checkpoints'][str(t)]:.3f}" for t in (300, 600, 900, 1800)]
    lines.append(f"| {r['group']}/{r['seed']} | " + " | ".join(values) + " |")
lines += ["", "## 候选质量（各组内合并计数）", "",
    "| 方法/来源 | 进入处理的提案数 | 处理前可行率 | 平均修复位数 | 修复后重复率 | 完整评估数 | 完整调度可行率 | 有效改进率 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|"]
def ratio(a, b):
    return f"{100*a/b:.2f}%" if b else "—"
for group in names:
    for origin in ("sa", "llm", "rule"):
        stages = [r["candidate_stages"][origin] for r in rows if r["group"] == group and origin in r["candidate_stages"]]
        if not stages:
            continue
        total = lambda k: sum(s[k] for s in stages)
        bits = [b for s in stages for b in s["repair_bits"]]
        mean = f"{statistics.mean(bits):.2f}" if bits else "—"
        lines.append(f"| {group}/{origin} | {total('raw_count')} | {ratio(total('raw_feasible_count'), total('raw_count'))} | {mean} | {ratio(total('repaired_duplicates'), total('repaired_count'))} | {total('full_evaluated')} | {ratio(total('full_feasible'), total('full_evaluated'))} | {ratio(total('improving_count'), total('full_evaluated'))} |")
lines += ["", "上表以实际进入提案处理记录的候选为分母；SA 在截止后返回的样本以及来不及进入修复的样本另列如下。规则为了获得四个不同候选而尝试的失败或重复修复均计入；完整评估列只统计该来源在预算内实际完成的不同方案。", "",
    "| 方法 | SA 返回样本总数 | 返回样本原始可行率 | 截止后返回样本数 | 进入修复的样本数 |",
    "|---|---:|---:|---:|---:|"]
for group in ("A", "B", "D", "E"):
    runs = [r for r in rows if r["group"] == group]
    returned = sum(r["sa_returned_raw"] for r in runs)
    feasible = sum(r["sa_returned_raw_feasible"] for r in runs)
    late = sum(r["sa_returned_after_deadline"] for r in runs)
    processed = sum(r["candidate_stages"].get("sa", {}).get("raw_count", 0) for r in runs)
    lines.append(f"| {group} | {returned} | {ratio(feasible, returned)} | {late} | {processed} |")
lines += ["",
    "## 时间分项（每组中位数，秒）", "",
    "| 方法 | QUBO 构建 | SA | 上下文 | LLM 等待及解析 | 规则计算 | 修复 | 块级调度 | 完整调度 | 物理复核 | 超时开销 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for group in names:
    runs = [r for r in rows if r["group"] == group]
    if not runs:
        continue
    values = []
    for field in ("qubo", "annealing", "context", "llm", "heuristic", "repair", "local_dispatch", "subproblems", "validation"):
        numbers = [r["timings"][field] for r in runs if field in r["timings"]]
        values.append(f"{statistics.median(numbers):.2f}" if numbers else "—")
    overshoots = [r["time_limit_overshoot"] for r in runs if r["time_limit_overshoot"] is not None]
    values.append(f"{statistics.median(overshoots):.2f}" if overshoots else "—")
    lines.append(f"| {group} | " + " | ".join(values) + " |")
lines += ["", "## 模型交接明细（全部尝试）", "",
    "认领至文件生成包含模型、工具及运行环境开销，不能视为纯模型推理时间；无回执或未被消费的输出不计分。", "",
    "| 调用目录 | 请求至认领/秒 | 认领至文件/秒 | 文件至回执/秒 | 求解器接收状态 |",
    "|---|---:|---:|---:|---|"]
for h in summary.get("handoff_timings", []):
    values = ["—" if h[k] is None else f"{h[k]:.2f}" for k in
              ("dispatch_delay_seconds", "file_ready_after_claim_seconds", "receipt_delay_after_file_seconds")]
    lines.append(f"| {h['call']} | " + " | ".join(values) + f" | {h['consumer_status'] or '未接收'} |")
lines += ["", "模型 token 用量未知。LLM 分项含文件交接和调度等待，不能解释为纯模型推理耗时。各分项中位数不保证相加等于总耗时中位数。", "",
    "请求认领及结果回执存在可观、变化较大的调度延迟；逐调用拆分见 final-summary.json 的 handoff_timings。C／27 第三次及 D／7 首次输出未在截止前完成交接验证，未计入成绩。",
    "原 C／17 的请求约等待 1507 秒才被认领，并且没有输出或回执；其 37.8906% 参考差只是初始解，不能作为正常 LLM 能力测量。",
    "补跑位于预定顺序之外，补齐运行结束记录，但不保证全部模型调用成功，也不消除调度与时间顺序带来的混杂因素。",
    "因此本批次的达标时间与截止成本反映实际端到端流程，不能把跨组时间差或调度导致的调用超时直接解释为 SA 协同或纯模型推理优势。", "",
    "## 推断范围", "", "是否支持协同，应结合 D 对 C/B/E 的配对差异、达标时间和 LLM 后 SA 的新增收益一起判断。",
    "E 是本协议规定的有限轮单块规则启发式，只能排除这一具体对照；不能据此断言所有传统启发式均不能替代 LLM。",
    "A/B/D/E 的搜索时限相同；C 在模型调用额度耗尽后提前结束，曲线延续其最后有效成本。",
    "参考求解独立进行，搜索没有使用其解。参考成本 885066.704431，下界 885066.704430；求解器耗时 95.679 秒，建模与求解合计 108.511 秒，独立调度复核另用 6.019 秒。",
    "QUBO 构建时间包含当前实现的开销，不能将该配置下的弱 SA 结果外推为 SA 或经典优化方法的一般上限；本批次不能支持相对独立 Gurobi 参考的加速主张。",
    "三个重复用于筛查，不足以支持统计显著性或跨实例结论；后续消融是否值得开展需依本批次结果决定。", "",
    "下一阶段应先修复协调会话中断时的模型请求与完成回执交接，另行登记补齐 C／17、D／7 等缺失的有效重复；再决定是否开展上下文、反馈与割更新消融。不能通过事后接收迟到输出、删除失败尝试或把等待时间从既定预算中扣除来改变本批次成绩。", "",
    "![最佳成本曲线](quality-curves.png)", "", "![近参考区域放大图](quality-detail.png)"]
(ROOT / "analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
