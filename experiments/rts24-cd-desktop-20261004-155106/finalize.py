"""Post-run verification and transparent handoff limitations; no model calls."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "orchestration"))
import batch


def main():
    assert batch.read(ROOT / "state.json")["status"] == "completed"
    batch.check_frozen()
    summary = batch.read(ROOT / "summary.json")
    assert summary["complete"] and summary["numerical_audit_passed"]
    manifest = batch.read(ROOT / "manifest.json")
    assert batch.digest(ROOT / "historical-summary.json") == manifest["historical_summary_sha256"]
    assert batch.digest(batch.OLD / "reference/result.json") == manifest["reference_sha256"]
    for name, expected in manifest["source_sha256"].items():
        assert batch.digest(batch.OLD / "snapshot" / name) == expected
    calls = []
    for group, seed in batch.ORDER:
        run = ROOT / f"{group}-{seed}"
        origin = batch.read(run / "run.json")["created_at"]
        for path in sorted((run / "calls").glob("*/request.json")):
            call = path.parent
            request = batch.read(path)
            launch = batch.read(call / "agent-launch.json")
            completion = batch.read(call / "agent-completion.json")
            receipt = batch.read(call / "receipt.json")
            consumed = batch.read(call / "consumed.json")
            output = call / "candidates.json"
            assert launch["transport"] == "collaboration.spawn_agent"
            assert completion["agent_id"] == launch["agent_id"]
            if consumed.get("status") == "completed":
                assert receipt["status"] == "completed" and completion["actual_turn_completed"]
            calls.append(dict(run=run.name,call=call.name,agent_id=launch["agent_id"],
                request_at=request["created_at"], launch_recorded_at=launch["launched_at"],
                candidate_file_mtime=output.stat().st_mtime if output.exists() else None,
                file_written_seconds_after_run_creation=output.stat().st_mtime-origin if output.exists() else None,
                completion_recorded_at=completion["finished_at"],
                file_to_completion_record_seconds=completion["finished_at"]-output.stat().st_mtime if output.exists() else None,
                receipt_published=completion["receipt_published"], consumed_status=consumed.get("status")))
    ids = [c["agent_id"] for c in calls]
    assert len(ids) == len(set(ids)) == summary["model_sessions"] <= 18
    import psutil
    owned = [p.info["pid"] for p in psutil.process_iter(["pid", "name", "cmdline"])
        if p.info["pid"] != os.getpid() and (p.info["name"] or "").lower().startswith("python")
        and any(str(ROOT).lower() in arg.lower() for arg in (p.info["cmdline"] or []))]
    assert not owned, owned
    verification = dict(verified_at=time.time(), complete=summary["complete"],
        numerical_audit_passed=summary["numerical_audit_passed"],
        llm_response_replications_complete=summary["llm_response_replications_complete"],
        desktop_model_sessions=len(calls), independent_codex_exec_model_sessions=0,
        owned_worker_processes_alive=owned,
        fresh_agent_ids_unique=True, original_snapshot_and_reference_unchanged=True,
        physical_evaluations_checked=sum(r["audit"]["physical_checked"] for r in summary["runs"]),
        monitoring_requirement_met=False,
        monitoring_limitations=["D/17 coordinator turn was interrupted; receipt was not delivered before deadline.",
            "C/7 candidate file existed before deadline, but completion handling occurred after cutoff; notification/coordination delay not isolated.",
            "Continuous minute-by-minute monitoring was not maintained for the entire batch; late agent termination and handoff delays remain in the record."],
        timestamp_note="File mtime is not evidence that the model turn had completed. launch_recorded_at is coordinator bookkeeping after spawn returned.",
        calls=calls, finalizer_sha256=batch.digest(Path(__file__)))
    batch.write(ROOT / "final-verification.json",verification)
    rows = sorted(summary["runs"],key=lambda r:(r["group"],r["seed"]))
    notes = ["# C/D 桌面端重跑结果与限制", "",
        f"六次预定实验已结束；数值审计通过；模型会话 {len(calls)}/18 次，截止前完整接收 {summary['successful_responses']} 次。全部结果保留，未自动重跑或补计迟到输出。", "",
        "| 方法/种子 | 最佳成本 | 参考差 | 完整响应 | LLM 后 SA/完整评估 | SA 新增降本 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        post=r["post_llm_sa"]
        notes.append(f"| {r['group']}/{r['seed']} | {r['objective']:.6f} | {100*r['reference_difference']:.4f}% | {r['llm_responses_consumed']}/{r['llm_started']} | {post['calls']}/{post['complete_evaluations']} | {r['sa_improvement_after_llm']:.6f} |")
    notes += ["", "D/17 在主协调回合中断期间到期；C/7 的候选虽在截止前写入，但完成事件处理及回执记录晚于截止，具体通知与协调延迟未进一步区分。两次均无模型候选计分，不能用于判断候选质量。文件存在也不能代替真实完成事件。",
        "", "本批未全程达到每分钟持续监测要求。已保留实际截止、交接失败及晚终止记录，不能将本批描述为无故障对照，也不能据此识别桌面端与后台调用的速度差异。",
        "", "数值审计只说明预算内计分、物理约束、成本、哈希和协议记录通过核对，不表示六次各有三次响应。当前数据不足以证明 SA 的必要性、双向协同、统计显著性或跨实例优势。",
        "", "详细结果见 [report.md](report.md)、[summary.json](summary.json)、[curves.csv](curves.csv)；最终复核见 [final-verification.json](final-verification.json)，文件校验见 [SHA256SUMS.txt](SHA256SUMS.txt)。",
        "", "原算法和通用 CLI 未修改；批次管理和审计代码位于本目录。25 项运行前测试通过，见 protocol-regression.log 和 batch-regression.log。", ""]
    (ROOT / "README.md").write_text("\n".join(notes),encoding="utf-8")
    report = ROOT / "report.md"
    text = report.read_text(encoding="utf-8")
    notice = "**交接与监测限制：D/17、C/7 的模型输出未在截止前完成回执交接，均未计分；本批未全程达到每分钟持续监测要求。数值审计通过不代表无故障比较，详见 [结果与限制](README.md)。**\n\n"
    if notice not in text:
        report.write_text(text.replace("\n\n", "\n\n"+notice,1),encoding="utf-8")
    print(json.dumps({k:v for k,v in verification.items() if k != "calls"},ensure_ascii=False))


if __name__ == "__main__":
    main()
