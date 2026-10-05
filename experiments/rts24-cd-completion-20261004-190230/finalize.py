"""Merge prespecified reruns and verify the previous archive was preserved."""
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "orchestration"))
import batch


def main():
    manifest = batch.check_frozen()
    current = batch.read(ROOT / "summary.json")
    assert current["complete"] and current["numerical_audit_passed"]
    previous = ROOT.parent / "rts24-cd-desktop-20261004-155106"
    checked = 0
    for line in (previous / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ",1)
        assert batch.digest(previous / name) == expected, name
        checked += 1
    old = batch.read(ROOT / "historical-summary.json")
    assert batch.digest(ROOT / "historical-summary.json") == manifest["historical_summary_sha256"]
    selected = {(r["group"],r["seed"]):dict(r,run_directory=str(previous / f"{r['group']}-{r['seed']}"),rerun=False) for r in old["runs"]}
    for row in current["runs"]:
        selected[row["group"],row["seed"]] = dict(row,run_directory=str(ROOT / f"{row['group']}-{row['seed']}"),rerun=True)
    rows = [selected[k] for k in sorted(selected)]
    assert set(selected) == {(g,s) for g in ("C","D") for s in (7,17,27)}
    numeric = all(r["audit"]["complete"] and r["audit"]["protocol_ok"] and not r["audit"]["errors"] and r["audit"]["result_objective_consistent"] for r in rows)
    full = all(r["llm_responses_consumed"] == 3 for r in rows)
    launches = [batch.read(p) for p in ROOT.glob("[CD]-*/calls/*/agent-launch.json")]
    assert len(launches) <= 9 and len({a["agent_id"] for a in launches}) == len(launches)
    assert all(a["transport"] == "collaboration.spawn_agent" for a in launches)
    monitor = batch.read(ROOT / "monitor-state.json")
    assert monitor["stopped"]
    result = dict(complete=True,numerical_audit_passed=numeric,llm_response_replications_complete=full,
        selected_runs=6,retained_attempts=6+len(current["runs"]),runs=rows,
        new_model_sessions=len(launches),new_responses=current["successful_responses"],
        original_files_verified=checked,original_unchanged=True,reference_cost=current["reference_cost"],
        monitor=monitor,selection="Replace D/27, D/17, C/7 irrespective of costs; retain all original attempts.",
        audited_at=time.time(),finalizer_sha256=batch.digest(Path(__file__)))
    batch.write(ROOT / "combined-summary.json",result)
    batch.write(ROOT / "final-verification.json",{k:v for k,v in result.items() if k != "runs"})
    with (ROOT / "combined-curves.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["group","seed","rerun","seconds","objective","source","gain"])
        writer.writeheader()
        for row in rows:
            writer.writerows(dict(group=row["group"],seed=row["seed"],rerun=row["rerun"],**point) for point in row["curve"])
    lines=["# C/D 缺失重复补跑", "",f"三个补跑已结束；合并六次的数值审计：{numeric}；六次均收齐三次响应：{full}。新增模型会话 {len(launches)}/9 次。", "",
        "每次 1800 秒，最多三次 Astra/max 全新桌面端子代理调用、每次最多四个候选；等待和交接计入预算。仅补跑 D/27、D/17、C/7，按预先指定结果替换，不按成本筛选。", "",
        "| 方法/种子 | 本次补跑 | 最佳成本 | 参考差 | 完整响应/请求 | LLM 后 SA/完整评估 | SA 新增降本 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        post=row["post_llm_sa"]
        lines.append(f"| {row['group']}/{row['seed']} | {'是' if row['rerun'] else '否'} | {row['objective']:.6f} | {100*row['reference_difference']:.4f}% | {row['llm_responses_consumed']}/{row['llm_started']} | {post['calls']}/{post['complete_evaluations']} | {row['sa_improvement_after_llm']:.6f} |")
    lines += ["", "本批只改变桌面端协调和本地连续观测，不改变搜索算法。监测每两秒记录一次状态，文件写出不等于模型完成；仍须桌面端真实完成事件后签发回执。", "",
        f"保留前批全部 273 个清单文件并复核 {checked} 个哈希。前批失败仍保留；本批与前批的时刻不同，本地种子不控制 LLM 随机性，配对不构成统计显著性或跨实例结论。", "",
        "合并数据见 [combined-summary.json](combined-summary.json)、[combined-curves.csv](combined-curves.csv)；本次三项详细候选和耗时见 [report.md](report.md)；复核见 [final-verification.json](final-verification.json)。", "", "![六次最终分析集](combined-quality-curves.png)", ""]
    (ROOT / "README.md").write_text("\n".join(lines),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for row in rows:
        points=row["curve"]
        xs=[p["seconds"] for p in points]+[1800]
        ys=[100*(p["objective"]/current["reference_cost"]-1) for p in points]
        ys += [ys[-1]]
        for ax in axes:
            ax.step(xs,ys,where="post",label=f"{row['group']}/{row['seed']}")
    for ax in axes:
        ax.set(xlabel="Elapsed seconds",ylabel="Reference cost difference (%)",xlim=(0,1800))
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_title("All selected runs")
    axes[1].set(title="Near-reference detail",ylim=(0,1.5))
    fig.tight_layout()
    fig.savefig(ROOT / "combined-quality-curves.png",dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k != "runs"},ensure_ascii=False))


if __name__ == "__main__":
    main()
