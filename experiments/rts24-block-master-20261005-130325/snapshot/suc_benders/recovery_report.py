"""Audit labelled recovery runs and compare them with the retained original batch."""
import argparse
import csv
import json
from pathlib import Path
import time

from .data import load_suc_data
from .minimal_report import summarize
from .rts24_experiment import read_events
from .subagent import publish, sha
from .validation import schedule_object, validate_schedule


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def audit_run(run, group, seed, data, scenarios, reference):
    events = read_events(run / "events.jsonl")
    result, state = read(run / "result.json"), read(run / "state.json")
    interruption = read(run / "interruption.json")
    if interruption:
        state = dict(state, status="INTERRUPTED", interruption=interruption)
    row = summarize(events, result, 1800, reference)
    errors, checked = [], 0
    for event in events:
        if event["kind"] == "evaluation" and event["feasible"] and event["elapsed_seconds"] <= 1800:
            try:
                validate_schedule(data, scenarios, schedule_object(event))
                checked += 1
            except (AssertionError, ValueError) as exc:
                errors.append(str(exc))
    received = 0
    handoffs = []
    for request_path in sorted((run / "calls").glob("*/request.json")):
        call = request_path.parent
        request, dispatch = read(request_path), read(call / "dispatch.json")
        response, consumed, agent = read(call / "receipt.json"), read(call / "consumed.json"), read(call / "agent-completion.json")
        if (request.get("model"), request.get("reasoning_effort"), request.get("fork_turns")) != ("gpt-6-astra", "max", "none"):
            errors.append(f"{call.name}: model/session configuration")
        if any(sha(call / name) != digest for name, digest in request["input_sha256"].items()):
            errors.append(f"{call.name}: input integrity")
        if response:
            if any(response.get(key) != request.get(key) for key in ("run_id", "call_id", "model", "reasoning_effort", "input_sha256")):
                errors.append(f"{call.name}: receipt identity")
            if response.get("status") == "completed":
                if (not (call / "candidates.json").exists() or not agent.get("actual_turn_completed") or agent.get("agent_id") != response.get("agent_id")
                        or sha(call / "candidates.json") != response.get("output_sha256")):
                    errors.append(f"{call.name}: completion/output integrity")
            if consumed.get("status") == "completed" and response.get("status") == "completed":
                received += 1
        output = call / "candidates.json"
        handoffs.append(dict(call=call.name, status=consumed.get("status"),
            receipt_status=response.get("status"), actual_turn_completed=agent.get("actual_turn_completed"),
            output_excluded=bool(output.exists() and consumed.get("status") != "completed"),
            dispatch_delay_seconds=dispatch.get("claimed_at", request["created_at"])-request["created_at"] if dispatch else None,
            receipt_delay_after_file_seconds=response["completed_at"]-output.stat().st_mtime if response and output.exists() else None,
            usage=agent.get("usage")))
    first = row["first_llm_improvement_seconds"]
    after = [e for e in events if first is not None and first < e["elapsed_seconds"] <= 1800]
    complete = state.get("status") in {"TIME_LIMIT", "LLM_CALL_LIMIT"} and bool(result)
    protocol_ok = (row["llm_started"] <= 3 and row["sa_seeds"] == [seed]*row["sa_calls"]
                   and (group != "C" or row["sa_calls"] == 0)
                   and all(e["diagnostics"].get("raw_count", 0) <= 4 for e in events if e["kind"] == "llm_end"))
    initial = next((e.get("commitment_hash") for e in events if e["kind"] == "evaluation"), None)
    row.update(group=group, seed=seed, state=state, llm_responses_consumed=received,
        llm_transport_status="all_three_received" if received == 3 else "partial" if received else "no_response",
        handoffs=handoffs, post_llm_sa=dict(calls=sum(e["kind"] == "sa_start" for e in after),
            complete_evaluations=sum(e["kind"] == "evaluation" and e["source"] == "sa" for e in after)),
        audit=dict(physical_checked=checked, errors=errors, complete=complete, protocol_ok=protocol_ok,
            initial_hash=initial, result_objective_consistent=bool(result and result["objective"] == row["objective"])))
    return row


def report(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    original = Path(manifest["original_directory"])
    old = read(original / "final-summary.json")
    data, scenarios = load_suc_data(directory / "snapshot/data/rts24/units.json", directory / "snapshot/data/rts24/stochastic.json")
    original_unchanged = all((original / name).exists() and sha(original / name) == digest
                             for name, digest in manifest["original_file_sha256"].items())
    source_ok = all(sha(directory / "snapshot" / name) == digest for name, digest in manifest["source_sha256"].items())
    reference_ok = (sha(original / "reference/result.json") == manifest["reference_sha256"]
                    and read(original / "reference/audit.json").get("feasible_solution"))
    selected = [dict(row, run_directory=str(original / row["run_directory"]), recovery=False) for row in old["runs"]]
    recoveries = []
    for group, seed in manifest["order"]:
        run = directory / f"{group}-{seed}"
        if not run.exists():
            continue
        row = audit_run(run, group, seed, data, scenarios, old["reference_cost"])
        row.update(run_directory=str(run), recovery=True)
        publish(run / "summary.json", row)
        recoveries.append(row)
        selected = [r for r in selected if (r["group"], r["seed"]) != (group, seed)] + [row]
    complete = read(directory / "state.json").get("status") == "completed" and len(recoveries) == 4
    started = sum(r["llm_started"] for r in recoveries)
    failed_startup_calls = manifest.get("failed_startup_calls", 0)
    interrupted_calls = manifest.get("interrupted_recovery_calls", 0)
    prior_unserved = manifest.get("prior_unserved_requests", 0)
    prior_unchanged = all((Path(folder) / name).exists() and sha(Path(folder) / name) == digest
        for folder, files in manifest.get("interrupted_recovery_sha256", {}).items() for name, digest in files.items())
    startup_unchanged = all((Path(folder) / name).exists() and sha(Path(folder) / name) == digest
        for folder, files in manifest.get("failed_startup_sha256", {}).items() for name, digest in files.items())
    model_sessions = len(list(directory.glob("*/calls/*/agent-launch.json")))
    total_recovery_sessions = failed_startup_calls + interrupted_calls + model_sessions
    authorized_limit = manifest.get("authorized_recovery_call_limit", 12)
    initial_hashes = {r["audit"]["initial_hash"] for r in selected}
    numerical_ok = bool(complete and original_unchanged and prior_unchanged and startup_unchanged and source_ok and reference_ok and old["passed"]
        and started <= 12 and len(selected) == 15 and len(initial_hashes) == 1 and None not in initial_hashes
        and total_recovery_sessions <= authorized_limit
        and all(r["audit"]["complete"] and r["audit"]["protocol_ok"] and not r["audit"]["errors"]
                and r["audit"]["result_objective_consistent"] and r["audit"]["physical_checked"] > 0 for r in selected))
    llm_runs = [r for r in selected if r["group"] in {"C", "D"}]
    all_received = all(r["llm_responses_consumed"] == 3 for r in llm_runs)
    payload = dict(complete=complete, numerical_audit_passed=numerical_ok,
        llm_response_replications_complete=all_received, original_unchanged=original_unchanged,
        source_hashes_match=source_ok, reference_verified=bool(reference_ok),
        new_llm_started=started, new_call_limit=12,
        failed_startup_calls=failed_startup_calls,
        interrupted_recovery_calls=interrupted_calls,
        prior_interrupted_records_unchanged=prior_unchanged,
        prior_startup_records_unchanged=startup_unchanged,
        new_model_sessions=model_sessions, recovery_model_sessions=total_recovery_sessions,
        prior_unserved_requests=prior_unserved,
        recovery_llm_started=failed_startup_calls+interrupted_calls+prior_unserved+started,
        authorized_recovery_call_limit=authorized_limit,
        failed_startup_directory=manifest.get("failed_startup_directory"),
        total_llm_started=old["llm_started"]+failed_startup_calls+interrupted_calls+prior_unserved+started,
        attempts_retained=old["attempted_runs"]+len(recoveries)+manifest.get("failed_startup_attempts",1 if failed_startup_calls else 0)
            +manifest.get("interrupted_recovery_attempts",0),
        selected_llm_runs_all_three_received=sum(r["llm_responses_consumed"] == 3 for r in llm_runs),
        paired_seeds_all_three_received=[seed for seed in (7, 17, 27) if
            all(any(r["group"] == group and r["seed"] == seed and r["llm_responses_consumed"] == 3
                    for r in llm_runs) for group in ("C", "D"))],
        reference_cost=old["reference_cost"], reference_lower_bound=old["reference_lower_bound"],
        protocol_change=manifest["protocol_change"], runs=sorted(selected, key=lambda r:(r["group"],r["seed"])),
        recovery_runs=recoveries, original_summary=str(original / "final-summary.json"), audited_at=time.time())
    publish(directory / "summary.json", payload)
    with (directory / "curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "seed", "recovery", "seconds", "objective", "source", "gain"])
        writer.writeheader()
        for row in selected:
            writer.writerows(dict(group=row["group"], seed=row["seed"], recovery=row["recovery"], **p) for p in row["curve"])
    number = lambda v: "—" if v is None else f"{v:.3f}"
    lines = ["# RTS-24 中断实验补跑（2026-10-04）", "",
        f"四项补跑结束：{complete}；数值审计通过：{numerical_ok}；六次 C/D 均收到三次响应：{all_received}。", "",
        f"保留全部 {payload['attempts_retained']} 次尝试；本批请求 {started}/12 次、实际启动 {model_sessions} 次模型会话；另保留 {failed_startup_calls} 次启动/认证失败及上批 {interrupted_calls} 次调用。累计恢复会话 {total_recovery_sessions}/{authorized_limit} 次。",
        f"另有此前 {prior_unserved} 次未派发的请求计入请求统计，不计入模型会话额度；全部批次累计请求 {payload['total_llm_started']} 次。",
        "事先指定补跑 C/17、D/7、C/27、D/17，按此顺序串行执行；对应结果全部纳入分析，不按成本挑选。",
        "算法、数据和依赖沿用原冻结快照；每次 1800 秒，从初始化计时。模型为 Astra/max，每次最多三次全新会话，每次至多四个候选。",
        "本次仅将模型调度改为独立后台 codex exec。等待、失败及交接仍计入预算，迟到输出不计分。旧尝试与归档未覆盖。",
        "新旧调度环境不同，跨日期时刻成绩存在混杂；下面的配对仅按本地种子，不控制 LLM 随机性。", "",
        "| 方法/种子 | 补跑 | 最佳成本 | 参考差 | 请求/接收 | ≤1% 秒 | ≤0.5% 秒 | LLM 后 SA 次数/完整评估 | LLM 后 SA 降本 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["runs"]:
        post = row.get("post_llm_sa", {})
        gap = "—" if row["reference_difference"] is None else f"{100*row['reference_difference']:.4f}%"
        lines.append(f"| {row['group']}/{row['seed']} | {'是' if row['recovery'] else '否'} | {number(row['objective'])} | {gap} | "
            f"{row['llm_started']}/{row.get('llm_responses_consumed',0)} | {number(row['threshold_seconds']['0.01'])} | "
            f"{number(row['threshold_seconds']['0.005'])} | {post.get('calls',0)}/{post.get('complete_evaluations',0)} | {number(row['sa_improvement_after_llm'])} |")
    lines += ["", "## 配对差异与证据范围", "", "D 减去 C，负数表示 D 成本更低。", "",
              "| 种子 | D − C | C/D 均接收三次响应 |", "|---|---:|---|"]
    by_pair = {(r["group"],r["seed"]):r for r in selected}
    for seed in (7,17,27):
        c,d = by_pair["C",seed],by_pair["D",seed]
        delta = d["objective"]-c["objective"] if c["objective"] is not None and d["objective"] is not None else None
        lines.append(f"| {seed} | {number(delta)} | {seed in payload['paired_seeds_all_three_received']} |")
    gains = sum(r["sa_improvement_after_llm"] for r in selected if r["group"] == "D")
    lines += ["", f"D 组首次 LLM 改善之后的 SA 新增降本合计：{gains:.3f}。",
        "若该值为零，当前实现仍未观察到 LLM 后 SA 的成本贡献；D 对 C 的差异还受候选输入、调用时机和随机输出影响。",
        "三次重复不足以证明统计显著性、跨实例优势或双向协同。独立 Gurobi 参考的求解时间仍约 96 秒，当前不能据此宣称加速。", "",
        "已完成评估的候选物理约束、成本、修复、重复、完整调度、分项耗时和截止检查见 summary.json；曲线见 curves.csv。未完成会话留下的候选文件不计分。",
        "模型会话及真实完成事件见各 calls 目录的 agent-events.jsonl；回执仅在进程成功退出、真实完成事件和候选文件均存在时签发。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    result = report(parser.parse_args().directory)
    print(json.dumps({k:result[k] for k in ("complete","numerical_audit_passed","llm_response_replications_complete","new_llm_started")}))
