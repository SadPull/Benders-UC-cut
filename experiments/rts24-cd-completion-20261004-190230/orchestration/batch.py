"""One frozen C/D batch; desktop Codex owns all model dispatches."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
OLD = REPO / "experiments/rts24-minimal-20261003"
sys.path.insert(0, str(ROOT / "snapshot"))
ORDER = [("D", 27), ("D", 17), ("C", 7)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (ROOT / "manifest.json").exists():
        raise ValueError("batch already prepared")
    old = read(OLD / "manifest.json")
    packages = {k: importlib.metadata.version(k) for k in old["packages"]}
    assert packages == old["packages"], (packages, old["packages"])
    assert Path(sys.executable).resolve() == Path(old["executable"]).resolve()
    for name, expected in old["source_sha256"].items():
        source = OLD / "snapshot" / name
        assert digest(source) == expected, name
        target = ROOT / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(OLD / "reference", ROOT / "reference")
    support = (REPO / "suc_benders/recovery_report.py").read_text(encoding="utf-8")
    (HERE / "audit_support.py").write_text(support.replace("from .", "from suc_benders."), encoding="utf-8")
    previous = REPO / "experiments/rts24-cd-desktop-20261004-155106/summary.json"
    shutil.copy2(previous, ROOT / "historical-summary.json")
    manifest = dict(created_at=time.time(), order=ORDER, budget_seconds=1800,
        groups={"C":"LLM-only", "D":"SA + LLM"}, seeds=[7,17,27], serial=True,
        model="gpt-6-astra", reasoning_effort="max", fork_turns="none",
        max_llm_calls_per_run=3, max_llm_calls_batch=9, max_candidates_per_call=4,
        source_sha256=old["source_sha256"], packages=packages, python=sys.version, executable=sys.executable,
        original_directory=str(OLD), reference_sha256=digest(ROOT / "reference/result.json"),
        reference_audit_sha256=digest(ROOT / "reference/audit.json"),
        historical_summary_sha256=digest(previous),
        protocol=dict(transport="desktop collaboration.spawn_agent", fresh_sessions=True,
            initialization_wait_handoff_validation_in_budget=True, late_candidates_excluded=True,
            reference_not_search_input=True, target_stopping=False, no_automatic_retries=True),
        authorization="User requested filling missing repetitions: rerun D/27, D/17, C/7 with original 1800-second budget and at most 9 fresh Astra/max desktop sessions.")
    write(ROOT / "manifest.json", manifest)
    write(ROOT / "state.json", dict(status="prepared", completed=[]))
    print(json.dumps(dict(directory=str(ROOT), order=ORDER, packages=packages)))


def check_frozen():
    manifest = read(ROOT / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / "snapshot" / name) == expected, name
    assert digest(ROOT / "reference/result.json") == manifest["reference_sha256"]
    assert digest(ROOT / "reference/audit.json") == manifest["reference_audit_sha256"]
    for name, expected in manifest.get("orchestration_sha256", {}).items():
        assert digest(HERE / name) == expected, name
    return manifest


def seal():
    manifest = check_frozen()
    manifest["orchestration_sha256"] = {p.name:digest(p) for p in HERE.glob("*.py")}
    write(ROOT / "manifest.json", manifest)
    print("orchestration sealed")


def alive(state):
    import psutil
    try:
        p = psutil.Process(state["worker_pid"])
        return abs(p.create_time()-state["worker_created_at"]) < 1 and p.is_running()
    except (psutil.NoSuchProcess, KeyError):
        return False


def start():
    import psutil
    from suc_benders.minimal_experiment import configuration
    from suc_benders.subagent import initialize
    manifest = check_frozen()
    state = read(ROOT / "state.json")
    assert state["status"] in {"prepared", "between_runs"}, state
    completed = state.get("completed", [])
    assert len(completed) < len(ORDER)
    group, seed = ORDER[len(completed)]
    run = ROOT / f"{group}-{seed}"
    initialize(run, dict(group=group, seed=seed, configuration=asdict(configuration(group, seed, run))))
    env = os.environ.copy()
    env.update({k:"1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-u", "-m", "suc_benders.minimal_experiment", "--worker", group,
               "--seed", str(seed), "--output-dir", str(run), "--rule-max-calls", "3"]
    with (run / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT / "snapshot", env=env, stdout=log,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    state.update(status="running", current=run.name, worker_pid=process.pid,
        worker_created_at=psutil.Process(process.pid).create_time(), started_at=time.time(),
        start_clock=time.perf_counter(), command=command)
    write(ROOT / "state.json", state)
    print(json.dumps(state))


def status():
    state = read(ROOT / "state.json")
    if state.get("status") != "running":
        return state
    run = ROOT / state["current"]
    ready = read(run / "ready.json")
    checkpoint = read(run / "best-checkpoint.json")
    progress = read(run / "progress.json")
    deadline = ready.get("deadline_clock", state["start_clock"]+60)
    calls = []
    for request_path in sorted((run / "calls").glob("*/request.json")):
        call = request_path.parent
        closed = (call / "consumed.json").exists() or (call / "receipt.json").exists()
        calls.append(dict(index=int(call.name), claimed=(call / "dispatch.json").exists(), closed=closed,
            agent=read(call / "agent-launch.json").get("agent_id"),
            completion=read(call / "agent-completion.json").get("actual_turn_completed"),
            consumed=read(call / "consumed.json").get("status")))
    return dict(status=state["status"], current=run.name, alive=alive(state),
        worker_state=read(run / "state.json"), remaining_seconds=round(deadline-time.perf_counter(), 2),
        progress={k:progress.get(k) for k in ("kind", "iteration", "call_index", "elapsed_seconds")},
        best=checkpoint.get("objective"), calls=calls)


def watch(seconds):
    until = time.perf_counter()+seconds
    while True:
        current = status()
        pending = any(not c["claimed"] and not c["closed"] for c in current.get("calls", []))
        if current.get("status") != "running" or not current.get("alive") or pending or current.get("remaining_seconds", 0) <= 0 or time.perf_counter() >= until:
            print(json.dumps(current, ensure_ascii=False))
            return
        time.sleep(.5)


def claim_call(index):
    from suc_benders.subagent import claim
    state = read(ROOT / "state.json")
    assert status()["remaining_seconds"] > 0
    launches = list(ROOT.glob("[CD]-*/calls/*/agent-launch.json"))
    assert len(launches) < 9
    print(claim(ROOT / state["current"], index))


def launched(index, agent):
    call = ROOT / read(ROOT / "state.json")["current"] / "calls" / f"{index:03d}"
    assert not (call / "agent-launch.json").exists()
    write(call / "agent-launch.json", dict(agent_id=agent, model="gpt-6-astra", reasoning_effort="max",
        fork_turns="none", launched_at=time.time(), transport="collaboration.spawn_agent"))
    print("launch recorded")


def finish(index, agent, outcome, message):
    from suc_benders.subagent import receipt
    state = read(ROOT / "state.json")
    run = ROOT / state["current"]
    call = run / "calls" / f"{index:03d}"
    assert read(call / "agent-launch.json")["agent_id"] == agent
    closed = (call / "consumed.json").exists() or (call / "receipt.json").exists() or status()["remaining_seconds"] <= 0
    (call / "agent-final.txt").write_text(message, encoding="utf-8")
    success = outcome == "completed"
    if success:
        assert (call / "candidates.json").exists()
    write(call / "agent-completion.json", dict(agent_id=agent, actual_turn_completed=success,
        status=outcome, evidence_source="desktop collaboration tool notification observed by coordinator",
        message=message, finished_at=time.time(), usage=None, late_or_closed=closed,
        receipt_published=not closed))
    if not closed:
        receipt(run, index, agent, "completed" if success else "failed")
    print(json.dumps(dict(receipt_published=not closed, status=outcome)))


def flags(rows, frozen_ok=True):
    complete = len(rows) == len(ORDER) and {(r["group"], r["seed"]) for r in rows} == set(ORDER) and all(r["audit"]["complete"] for r in rows)
    numerical = complete and frozen_ok and sum(r["llm_started"] for r in rows) <= 9 and all(
        r["audit"]["protocol_ok"] and not r["audit"]["errors"] and r["audit"]["result_objective_consistent"]
        and r["audit"]["physical_checked"] > 0 for r in rows)
    return dict(complete=complete, numerical_audit_passed=bool(numerical),
        llm_response_replications_complete=complete and all(r["llm_responses_consumed"] == 3 for r in rows))


def report():
    import csv
    from audit_support import audit_run
    from suc_benders.data import load_suc_data
    manifest = check_frozen()
    reference = read(ROOT / "reference/result.json")
    data, scenarios = load_suc_data(ROOT / "snapshot/data/rts24/units.json", ROOT / "snapshot/data/rts24/stochastic.json")
    rows = []
    for group, seed in ORDER:
        run = ROOT / f"{group}-{seed}"
        if not (run / "events.jsonl").exists():
            continue
        row = audit_run(run, group, seed, data, scenarios, reference["objective"])
        previous = None
        for request_path in sorted((run / "calls").glob("*/request.json")):
            call = request_path.parent
            user = read(call / "user.txt")
            context = user.get("context", {})
            if "previous_call_feedback" in context and context["previous_call_feedback"] != previous:
                row["audit"]["errors"].append(f"{call.name}: previous feedback mismatch")
            previous = read(call / "feedback.json") or None
        write(run / "summary.json", row)
        rows.append(row)
    checks = flags(rows, read(ROOT / "reference/audit.json").get("feasible_solution", False))
    initial = {r["audit"]["initial_hash"] for r in rows}
    if len(initial) != 1 or None in initial:
        checks["numerical_audit_passed"] = False
    payload = dict(**checks, expected_runs=len(ORDER), runs=rows, reference_cost=reference["objective"],
        reference_lower_bound=reference["lower_bound"], source_hashes_match=True,
        llm_requests=sum(r["llm_started"] for r in rows),
        model_sessions=len(list(ROOT.glob("[CD]-*/calls/*/agent-launch.json"))),
        successful_responses=sum(r["llm_responses_consumed"] for r in rows), audited_at=time.time())
    write(ROOT / "summary.json", payload)
    with (ROOT / "curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group","seed","seconds","objective","source","gain"])
        writer.writeheader()
        for r in rows:
            writer.writerows(dict(group=r["group"], seed=r["seed"], **p) for p in r["curve"])
    n = lambda x: "—" if x is None else f"{x:.3f}"
    lines = ["# C/D 桌面端子代理重跑", "", f"运行完成：{checks['complete']}；数值审计通过：{checks['numerical_audit_passed']}；三次补跑均收齐三次响应：{checks['llm_response_replications_complete']}。", "",
        "每次 1800 秒，从初始化计时；Astra/max、全新桌面端子代理、最多三次调用，每次最多四个候选。等待、交接及验证计入预算，迟到结果排除。",
        f"参考成本 {reference['objective']:.6f}，仅用于事后比较；参考差不是本算法认证的最优间隙。", "",
        "| 方法/种子 | 状态 | 最佳成本 | 参考差 | 请求/响应 | ≤1% 秒 | ≤0.5% 秒 | LLM 后 SA/完整评估 | SA 新增降本 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r:(r["group"],r["seed"])):
        gap = "—" if r["reference_difference"] is None else f"{100*r['reference_difference']:.4f}%"
        post = r["post_llm_sa"]
        lines.append(f"| {r['group']}/{r['seed']} | {r['state'].get('status')} | {n(r['objective'])} | {gap} | {r['llm_started']}/{r['llm_responses_consumed']} | {n(r['threshold_seconds']['0.01'])} | {n(r['threshold_seconds']['0.005'])} | {post['calls']}/{post['complete_evaluations']} | {n(r['sa_improvement_after_llm'])} |")
    lines += ["", "## 固定时刻成本", "", "| 方法/种子 | 300 秒 | 600 秒 | 900 秒 | 1800 秒 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['group']}/{r['seed']} | " + " | ".join(n(r["checkpoints"][str(t)]) for t in (300,600,900,1800))+" |")
    lines += ["", "## 候选质量", "", "| 方法/种子/来源 | 提案 | 原始离散可行率 | 平均修复位 | 修复后重复率 | 完整评估 | 调度可行率 | 改进率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for source, s in r["candidate_stages"].items():
            pct = lambda v:"—" if v is None else f"{100*v:.2f}%"
            lines.append(f"| {r['group']}/{r['seed']}/{source} | {s['raw_count']} | {pct(s['raw_feasible_rate'])} | {n(s['repair_bits_mean'])} | {pct(s['repaired_duplicate_rate'])} | {s['full_evaluated']} | {pct(s['full_feasible_rate'])} | {pct(s['improvement_rate'])} |")
    lines += ["", "## 耗时与交接", "", "分项耗时、每次派发延迟、文件至回执延迟见 summary.json 的 timings/handoffs。LLM 耗时包含桌面端协调与文件交接，未知 token 用量保留 null。", "",
        "| 方法/种子 | QUBO 秒 | SA 秒 | LLM 等待秒 | 完整调度秒 | 超时开销秒 |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['group']}/{r['seed']} | " + " | ".join(n(r["timings"].get(k)) for k in ("qubo","annealing","llm","subproblems"))+f" | {n(r['time_limit_overshoot'])} |")
    lines += ["", "## 本批配对与历史对照", "", "| 种子 | 本批 D−C 成本 | 双方均收齐三次响应 |", "|---|---:|---|"]
    pairs = {(r["group"],r["seed"]):r for r in rows}
    for seed in (7,17,27):
        c,d = pairs.get(("C",seed)),pairs.get(("D",seed))
        if c and d:
            delta = d["objective"]-c["objective"] if c["objective"] is not None and d["objective"] is not None else None
            lines.append(f"| {seed} | {n(delta)} | {c['llm_responses_consumed']==d['llm_responses_consumed']==3} |")
    lines += ["", "历史表来自紧接本批之前的桌面端六次实验；本报告单列三次补跑，最终合并表另见 README.md。", "", "| 方法/种子 | 历史成本 | 本批成本 |", "|---|---:|---:|"]
    for old in read(ROOT / "historical-summary.json")["runs"]:
        key = (old["group"],old["seed"])
        if key[0] in {"C","D"}:
            lines.append(f"| {key[0]}/{key[1]} | {n(old['objective'])} | {n(pairs.get(key,{}).get('objective'))} |")
    lines += ["", "三个种子只用于初步筛查，不能证明统计显著性或跨实例优势。D 没有后续 SA 时间与 SA 执行后没有收益分别解释。", "", "![成本曲线](quality-curves.png)", ""]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(12,4.5))
        for r in rows:
            points = r["curve"]
            if not points: continue
            xs = [p["seconds"] for p in points]+[1800]
            ys = [100*(p["objective"]/reference["objective"]-1) for p in points]
            ys += [ys[-1]]
            for ax in axes:
                ax.step(xs,ys,where="post",label=f"{r['group']}/{r['seed']}")
        for ax in axes:
            ax.set(xlabel="Elapsed seconds",ylabel="Reference cost difference (%)",xlim=(0,1800))
            ax.grid(alpha=.25)
            ax.legend()
        axes[1].set_ylim(0,1.5)
        axes[0].set_title("Full range")
        axes[1].set_title("Near-reference detail")
        fig.tight_layout()
        fig.savefig(ROOT / "quality-curves.png",dpi=160)
        plt.close(fig)
    print(json.dumps({k:payload[k] for k in checks}|dict(runs=len(rows),requests=payload["llm_requests"],responses=payload["successful_responses"])))
    return payload


def close_run():
    state = read(ROOT / "state.json")
    assert state["status"] == "running" and not alive(state)
    run = ROOT / state["current"]
    worker_state = read(run / "state.json")
    for launch in (run / "calls").glob("*/agent-launch.json"):
        assert (launch.parent / "agent-completion.json").exists(), "active/unrecorded agent"
    good = worker_state.get("status") in {"TIME_LIMIT", "LLM_CALL_LIMIT"} and (run / "result.json").exists()
    if not good:
        state.update(status="failed", failure=worker_state)
        write(ROOT / "state.json", state)
        report()
        raise RuntimeError("worker failed; batch stopped")
    state["completed"].append(state["current"])
    state["status"] = "completed" if len(state["completed"]) == len(ORDER) else "between_runs"
    write(ROOT / "state.json", state)
    payload = report()
    row = next(r for r in payload["runs"] if f"{r['group']}-{r['seed']}" == run.name)
    assert not row["audit"]["errors"] and row["audit"]["protocol_ok"] and row["audit"]["result_objective_consistent"], row["audit"]


def checksums():
    check_frozen()
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256SUMS.txt")
    (ROOT / "SHA256SUMS.txt").write_text("".join(f"{digest(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in paths),encoding="utf-8")
    print(json.dumps(dict(files=len(paths),state=read(ROOT / "state.json")["status"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","seal","start","status","watch","claim","launched","finish","close","report","checksums"])
    parser.add_argument("--index",type=int)
    parser.add_argument("--agent")
    parser.add_argument("--outcome", choices=["completed","failed","interrupted"])
    parser.add_argument("--message",default="")
    parser.add_argument("--seconds",type=float,default=30)
    args = parser.parse_args()
    if args.action == "status": print(json.dumps(status(),ensure_ascii=False))
    elif args.action == "watch": watch(args.seconds)
    elif args.action == "claim": claim_call(args.index)
    elif args.action == "launched": launched(args.index,args.agent)
    elif args.action == "finish": finish(args.index,args.agent,args.outcome,args.message)
    elif args.action == "close": close_run()
    else: globals()[args.action]()

