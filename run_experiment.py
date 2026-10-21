"""输入固定读取 data 中的 JSON；在 VS Code 中点击“运行 Python 文件”。"""
from datetime import datetime, timedelta, timezone
from dataclasses import fields
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

from run_suc import run_suc, _serializable
from suc_benders.anneal import SA_BACKEND, SA_INITIALIZATION
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model
from suc_benders.advisor_request import SessionUnavailable, session_file
from suc_benders.optimization import optimize
from suc_benders.validation import unpack, validate


# 默认算例为 data/；切换算例改 DATA_DIR。以下只设置运行参数。
SEED = 17                # SA 随机种子
TIME_LIMIT_SECONDS = 7200  # 本次运行预算；续跑时额外增加这些秒，包含恢复、LLM 和记录开销
RESUME_FROM = None       # 从零开始；续跑时填入解压后的完整 checkpoint.json.gz 路径
REFERENCE_TIME_LIMIT_SECONDS = 120  # 完整问题参考求解的独立预算，单位：秒
TARGET_GAP_PERCENT = 0.1  # 认证间隙目标，单位：%；0.1 就是 0.1%
CHECK_ONLY = False       # True：只检查输入；False：实际求解
LLM_ENABLED = True       # True：停滞时自动请求 LLM 建议（通道见 LLM_TRANSPORT）；False：关闭 LLM
LLM_PATIENCE = 1         # 连续一个完整 SA 轮次改善不足 0.05% 时触发
LLM_FIRST_CALL_AT_START = True  # True：初始化后立即发首个请求，不等待 SA 停滞
LLM_MODEL = "gpt-6-astra"  # 使用 Codex 已登录账号，推理档位为 max
LLM_TRANSPORT = "codex"  # "codex"：Codex 独立任务；"http"：流式 Responses，URL/密钥取自 config.local.json
LLM_TIMEOUT_SECONDS = 1800  # 剩余时间不足完整请求窗口加预留时，不启动新请求
LLM_RESERVE_SECONDS = 60  # 不用于等待 LLM，留给后续调度、校验和数值搜索
LLM_MAX_CALLS = None     # 全程调用上限；None：在预算内按停滞规则继续，保留累计调用计数
BLOCK_HOURS = 12         # 分块小时数；RCUC20 调参：3 h 块被长最小停机时间锁在块坐标局部最优
SA_SWEEPS = 16           # 每个温度对全部位的扫描次数；12 h 块下 16 优于默认 4
DATA_DIR = "data"        # 算例目录（相对项目根）；其他算例在 instances/<名称>

# Codex 提供上界候选，独立数值进程提供认证下界。
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def save(path, value):
    compressed = path.suffix == ".gz"
    with (gzip.open(path, "wt", encoding="utf-8") if compressed else path.open("w", encoding="utf-8")) as stream:
        stream.write(json.dumps(_serializable(value), ensure_ascii=False,
                                indent=None if compressed else 2, allow_nan=False))
        stream.write("\n")


def freeze_experiment(output, options, resume_source=None):
    paths = [PROJECT / name for name in ("run_experiment.py", "run_suc.py", "requirements.txt", "README.md")]
    paths += sorted((PROJECT / "suc_benders").glob("*.py"))
    paths += sorted((PROJECT / "tests").glob("*.py"))
    folder = Path(options.data_file).resolve().parent
    paths += sorted(folder.rglob("*.json"))
    paths += sorted(p for p in (folder / "sources").glob("*") if p.is_file())
    hashes = {}
    for source in paths:
        relative = source.relative_to(PROJECT)
        content = source.read_bytes()
        target = output / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    effective = {f.name: (getattr(options, f.name) if f.repr else
                         "<redacted>" if getattr(options, f.name) else "") for f in fields(options)}
    save(output / "effective-options.json", effective)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT, capture_output=True, text=True)
    save(output / "manifest.json", dict(
        created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        git_head=git_head.stdout.strip() if git_head.returncode == 0 else None,
        git_status=git_status.stdout if git_status.returncode == 0 else None,
        python=sys.version, executable=sys.executable,
        sa_backend=SA_BACKEND, sa_initialization=SA_INITIALIZATION,
        dependencies={name: version(name) for name in
                      ("numpy", "scipy", "gurobipy", "dwave-samplers", "dimod", "networkx", "tomli")},
        files=hashes, effective_options_sha256=hashlib.sha256(
            (output / "effective-options.json").read_bytes()).hexdigest(),
        cold_start=resume_source is None, inherited_evidence=resume_source is not None,
        resume_source=resume_source, reference_feedback=False))


def solution_snapshot(result, elapsed, prior=0., session_id=None):
    # shortcut: this saves the feasible solution, not a resumable search state; add recovery only when requested.
    values = {name: getattr(result, name) for name in
              ("status", "message", "objective", "lower_bound", "u", "v", "w", "scenarios", "validation", "llm_calls")}
    gap = max(0., result.objective-result.lower_bound)/max(1., abs(result.objective)) if np.isfinite(
        [result.objective, result.lower_bound]).all() else np.inf
    return dict(values, format="SOLUTION_SNAPSHOT_NOT_CHECKPOINT", relative_gap=gap,
                sa_backend=result.diagnostics.get("sa_backend"),
                sa_initialization=result.diagnostics.get("sa_initialization"),
                elapsed_seconds=elapsed, segment_elapsed_seconds=elapsed-prior, session_id=session_id,
                model_fingerprint=result.diagnostics.get("model_fingerprint"),
                sa_calls=result.diagnostics.get("sa_calls", 0))


# 精简一次 LLM 调用的回执：保留请求基准 base_u 与模型原样改动 raw_changes（二者可还原完整方案），
# 逐候选写出格式剔除原因、评估结果、调度状态与费用，以及移植结果；不保存可由此重建的整张矩阵。
# accepted 是评估后被接受的候选数，format_valid/invalid 是解析阶段的合格/剔除数。
def llm_record(row):
    record = {key: row.get(key) for key in (
        "call_id", "status", "thread_id", "requested_session_id", "request_directory", "background_seconds",
        "request_upper", "upper_before_probes", "upper_after_probes", "prompt_mode", "usage", "message",
        "accepted", "format_valid", "invalid", "base_u", "response_model", "response_output")}
    measured = {r["proposal_id"]: r for r in (row.get("feedback") or {}).get("proposals", [])}
    record["candidates"] = []
    for p in row.get("proposals") or []:
        m = measured.get(p.get("proposal_id"), {})
        record["candidates"].append(dict(
            proposal_id=p.get("proposal_id"), raw_changes=p.get("raw_changes"), invalid_reason=p.get("invalid_reason"),
            outcome=p.get("outcome"), repair_bits=p.get("hamming_u"),
            raw_constraint_violations=p.get("raw_constraint_violations"), feasible=m.get("feasible"),
            infeasibility_reason=m.get("infeasibility_reason"), reachable_shortfall=m.get("reachable_shortfall"),
            costs=m.get("costs"),
            cost_change=m.get("cost_change"), incumbent_decrease=p.get("incumbent_decrease")))
    record["rebase"] = [dict(proposal_id=r.get("proposal_id"), rebased_from=r.get("rebased_from"),
                             units=r.get("units"), outcome=r.get("outcome"), repair_bits=r.get("hamming_u"),
                             incumbent_decrease=r.get("incumbent_decrease"))
                        for r in row.get("rebase") or []]
    return record


class ExperimentProgress:
    def __init__(self, output, prior=0.):
        self.output = output
        self.prior = prior
        self.last_upper = np.inf
        self.session_id = None
        self.recording_seconds = 0.

    def __call__(self, whole, event):
        result = whole.result
        phase = perf_counter()
        elapsed = phase - whole.started
        improved = result.objective < self.last_upper
        if event in ("before_sampling", "sampling") or (event == "evaluation" and not improved):
            return
        self.last_upper = result.objective
        self.session_id = whole.advice.thread_id
        row = dict(event=event, elapsed_seconds=elapsed, upper=result.objective, lower=result.lower_bound,
                   sa_calls=whole.diagnostics["sa_calls"], llm_calls=result.llm_calls,
                   resume_cursor=whole.resume_cursor, status=result.status,
                   segment_elapsed_seconds=elapsed-self.prior, session_id=self.session_id)
        if event.startswith("llm") and result.llm_history:
            row["llm_status"] = result.llm_history[-1]["status"]
        if event == "coordination" and result.outer_history:
            row["coordination"] = result.outer_history[-1]["coordination"]
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_serializable(row), ensure_ascii=False, allow_nan=False) + "\n")
        if improved or event in ("initialization", "final"):
            temporary = self.output / "latest-solution.tmp.json"
            save(temporary, solution_snapshot(result, elapsed, self.prior, self.session_id))
            temporary.replace(self.output / "latest-solution.json")
        self.recording_seconds += perf_counter() - phase


def solve_reference(data, scenarios, seconds, log_file):
    started = perf_counter()
    full = model(data, scenarios)
    raw = optimize(full, seconds, integral=True, log_file=log_file)
    reference = dict(status=raw.status, objective=None, lower=raw.lower, gap_percent=None,
                     validation=None)
    if raw.x is not None:
        point = raw.x.copy()
        point[full.binary] = np.rint(point[full.binary])
        plan = unpack(full, point)
        if abs(plan.objective - raw.objval) > 1e-4:
            raise ValueError("参考方案重算成本与 Gurobi 目标值不一致")
        reference.update(vars(plan), validation=validate(data, scenarios, plan),
                         gap_percent=100 * max(0., plan.objective - raw.lower)
                         / max(1., abs(plan.objective)))
    reference["elapsed_seconds"] = perf_counter() - started
    return reference


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = RESULTS / f"{'resume' if RESUME_FROM else 'manual'}-seed{SEED}-{stamp}-{uuid4().hex[:8]}"
    folder = PROJECT / DATA_DIR
    if LLM_TRANSPORT not in ("codex", "http"):
        raise ValueError("LLM_TRANSPORT must be 'codex' or 'http'")
    codex = LLM_ENABLED and LLM_TRANSPORT == "codex"
    options = dict(data_file=str(folder / "units.json"), scenario_file=str(folder / "stochastic.json"),
                   seed=SEED, time_limit=TIME_LIMIT_SECONDS,
                   tolerance=TARGET_GAP_PERCENT / 100, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS, max_outer=1000000,
                   max_inner=20, llm_enabled=LLM_ENABLED, llm_model=LLM_MODEL, llm_patience=LLM_PATIENCE,
                   llm_max_calls=LLM_MAX_CALLS, llm_first_call_at_start=LLM_FIRST_CALL_AT_START,
                   llm_timeout=LLM_TIMEOUT_SECONDS, llm_reserve_seconds=LLM_RESERVE_SECONDS,
                   llm_codex_command="codex" if codex else "",
                   llm_agent_directory=str(output / "llm_requests") if codex else "")
    checked = load_options(options)
    data, scenarios = load_data(checked.data_file, checked.scenario_file)
    dimensions = dict(units=data.n_units, hours=data.horizon, scenarios=len(scenarios))
    print(f"机组输入：{checked.data_file}", flush=True)
    print(f"场景输入：{checked.scenario_file}", flush=True)
    print(f"输入规模：{data.n_units}机组；{data.horizon}小时；{len(scenarios)}场景", flush=True)
    if CHECK_ONLY:
        print("输入检查通过；本次没有执行求解或检查授权。")
        return 0

    resume_state, resume_source, prior, prior_calls = None, None, 0., 0
    if RESUME_FROM:
        source = (PROJECT / RESUME_FROM).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        audit_file = source.parent / "audit.json"
        if audit_file.exists():
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            if audit["status"] != "PASS" or audit["artifact_sha256"][source.name] != digest:
                raise ValueError("Source checkpoint audit/hash mismatch")
        manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
        for name in ("units.json", "stochastic.json"):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != manifest["files"][(folder/name).relative_to(PROJECT).as_posix()]:
                raise ValueError("Resume inputs differ from the source experiment")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            resume_state = json.load(stream)
        prior = resume_state["charged_elapsed_seconds"]
        if not np.isfinite(prior) or prior < 0:
            raise ValueError("Invalid cumulative elapsed time")
        prior_calls = resume_state["llm_calls"]
        session_id = next((row["thread_id"] for row in reversed(resume_state["llm_history"])
                           if row["status"] == "OK" and row.get("thread_id")), None)
        if LLM_ENABLED and prior_calls:
            try:
                session_file(session_id)
            except SessionUnavailable as error:
                print(f"续跑会话不可用（{error}）；首次请求失败后将新开会话。", flush=True)
        resume_source = dict(path=str(source), sha256=digest, charged_elapsed_seconds=prior,
                             initial_upper=resume_state["objective"], initial_lower=resume_state["lower_bound"],
                             prior_llm_calls=prior_calls, session_id=session_id)
        checked.time_limit = options["time_limit"] = prior + TIME_LIMIT_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    save(output / "options.json", options)
    freeze_experiment(output, checked, resume_source)
    progress = ExperimentProgress(output, prior)
    print(f"结果目录：{output}", flush=True)
    print(f"随机种子：{SEED}；主算法预算：{TIME_LIMIT_SECONDS} 秒；"
          f"认证间隙目标：{TARGET_GAP_PERCENT}%；LLM：{'开启' if LLM_ENABLED else '关闭'}", flush=True)
    if LLM_ENABLED:
        print(f"调用方式：{'Codex 独立任务' if codex else 'HTTP Responses 流式'}；模型：{checked.llm_model}；推理档位：max；"
              f"{'初始化后先发首个请求，之后' if checked.llm_first_call_at_start else ''}"
              f"连续 {checked.llm_patience} 个完整 SA 轮次改善不足时触发，"
              f"最多调用 {checked.llm_max_calls if checked.llm_max_calls is not None else '不限'} 次。", flush=True)
        if codex:
            print("复用 Codex 已有登录；请求和日志保存到 llm_requests；开启不代表立即调用。", flush=True)
            print(f"Codex 直接使用 HTTPS；单次最多等待 {checked.llm_timeout:g} 秒，"
                  f"保留 {checked.llm_reserve_seconds:g} 秒给后续数值处理。", flush=True)
        else:
            print(f"URL 与密钥取自 config.local.json；每次发送完整提示，用量和回复保存到 result.json；"
                  f"单次最多等待 {checked.llm_timeout:g} 秒，保留 {checked.llm_reserve_seconds:g} 秒给后续数值处理。",
                  flush=True)
    print("指标说明：UB 是通过全天物理校验的可行方案成本；LB 是认证下界；"
          "认证间隙 = (UB - LB) / max(1, |UB|)。", flush=True)
    print("块内上下界用于分块搜索，不等于全天原问题的上下界。", flush=True)
    if resume_source:
        print(f"续跑来源：{resume_source['path']}；继承 UB={resume_source['initial_upper']:.6f}；"
              f"LB={resume_source['initial_lower']:.6f}；已用 {prior:.2f} 秒；会话={resume_source['session_id']}", flush=True)
    print("开始主算法：恢复已有方案或初始化、分块 SA 搜索和独立强化下界认证。", flush=True)
    started = perf_counter()
    try:
        result = run_suc(vars(checked), resume_state=resume_state, progress=progress, started_at=started)
    except Exception as error:
        save(output / "failure.json", dict(status="INCOMPLETE", error_type=type(error).__name__,
                                          elapsed_seconds=perf_counter() - started))
        raise
    solver_wall = perf_counter() - started
    final_save_started = perf_counter()
    compact = solution_snapshot(result, result.elapsed_seconds, prior, progress.session_id)
    compact.update(timings=result.diagnostics.get("timings", {}),
                   search_statistics=result.diagnostics.get("search_statistics", {}),
                   certification_history=result.certification_history,
                   llm_history=[llm_record(row) for row in result.llm_history])
    save(output / "result.json", compact)
    final_save_seconds = perf_counter() - final_save_started
    summary = dict(scope="RESUMED_CONVERGENCE_CHECK" if resume_state is not None else "COLD_START_EXPLORATORY",
                   audit_status="NOT_REQUESTED", **dimensions, resume_source=resume_source,
                   seed=SEED, status=result.status, message=result.message,
                   sa_backend=result.diagnostics.get("sa_backend"),
                   sa_initialization=result.diagnostics.get("sa_initialization"),
                   upper=result.objective, lower=result.lower_bound,
                   gap_percent=100 * result.relative_gap, reference_gap_percent=None,
                   validation=result.validation,
                   solver_elapsed_seconds=result.elapsed_seconds,
                   segment_solver_elapsed_seconds=result.elapsed_seconds-prior,
                   segment_wall_seconds=solver_wall,
                   recording_seconds=progress.recording_seconds, final_save_seconds=final_save_seconds,
                   session_id=progress.session_id, new_llm_calls=result.llm_calls-prior_calls,
                   time_limit_overshoot_seconds=result.diagnostics.get("time_limit_overshoot", 0.),
                   wall_to_result_saved_seconds=perf_counter() - started,
                   sa_calls=result.diagnostics.get("sa_calls", 0), llm_calls=result.llm_calls,
                   block_sa_calls=result.diagnostics.get('block_sa_calls', 0),
                   llm_patience=LLM_PATIENCE, llm_first_call_at_start=LLM_FIRST_CALL_AT_START,
                   llm_disabled=result.diagnostics.get('llm_disabled'),
                   data_dir=DATA_DIR, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS,
                   dispatch_failures=result.diagnostics.get('dispatch_failures', 0),
                   search_statistics=result.diagnostics.get('search_statistics', {}),
                   note="轻量留证；latest-solution.json 仅保存方案，不是完整搜索检查点。"
                        "续跑不做参考求解或完整审计；没有自动补跑。")
    save(output / "summary.json", summary)
    status_text = {"CONVERGED": "已达到认证间隙目标", "TIME_LIMIT": "达到时间上限",
                   "MAX_ITERATIONS": "达到迭代上限"}.get(result.status, "求解未正常完成")
    print(f"\n主算法状态：{status_text}（{result.status}）", flush=True)
    if result.message:
        print(f"状态详情：{result.message}", flush=True)
    if result.validation is not None:
        print(f"主算法可行成本（UB）：{result.objective:.6f}", flush=True)
    else:
        print("主算法尚未获得通过校验的全天可行方案。", flush=True)
    print(f"认证下界（LB）：{result.lower_bound:.6f}；"
          f"认证间隙：{100 * result.relative_gap:.4f}%", flush=True)
    print(f"主算法耗时：{result.elapsed_seconds:.2f} 秒；"
          f"SA 调用：{summary['sa_calls']} 次；LLM 调用：{result.llm_calls} 次", flush=True)

    if resume_state is not None:
        print(f"本段运行 {solver_wall:.2f} 秒；累计调用 {result.llm_calls} 次，其中新增 {result.llm_calls-prior_calls} 次；"
              f"记录开销 {progress.recording_seconds:.3f} 秒；最终保存 {final_save_seconds:.3f} 秒。", flush=True)
        print(f"续跑结果已保存：{output}；本次不再求参考解或执行完整审计。", flush=True)
        return int(result.status.endswith("FAILED") or result.validation is None)

    print(f"\n[2/2] Gurobi 直接求解同一完整问题（保留全部整数变量、时段和场景），"
          f"另用最多 {REFERENCE_TIME_LIMIT_SECONDS} 秒。", flush=True)
    print("参考求解仅用于比较，其方案和下界不回填主算法；日志见 reference-gurobi.log。", flush=True)
    reference_started = perf_counter()
    try:
        reference = solve_reference(data, scenarios, REFERENCE_TIME_LIMIT_SECONDS, output / "reference-gurobi.log")
    except Exception as error:
        reference = dict(status="REFERENCE_FAILED", message=f"{type(error).__name__}: {error}",
                         objective=None, lower=None, gap_percent=None, validation=None,
                         elapsed_seconds=perf_counter() - reference_started)
    reference["time_limit_seconds"] = REFERENCE_TIME_LIMIT_SECONDS
    save(output / "reference.json", reference)
    summary["reference"] = {key: value for key, value in reference.items()
                            if key not in ("u", "v", "w", "scenarios")}
    if (result.validation is not None and reference["validation"] is not None
            and reference["objective"] != 0.):
        summary["reference_gap_percent"] = (100 * (result.objective - reference["objective"])
                                            / reference["objective"])
    save(output / "summary.json", summary)
    print(f"结果已保存：{output}；精简主结果见 result.json，参考方案见 reference.json。", flush=True)
    print(f"参考求解状态：{reference['status']}；耗时：{reference['elapsed_seconds']:.2f} 秒", flush=True)
    if reference["validation"] is not None:
        quality = "在求解器容差内最优" if reference["status"] == "OPTIMAL" else "可行参考方案，尚未证明最优"
        print(f"参考求解自身的最优间隙：{reference['gap_percent']:.6f}%", flush=True)
        print(f"Gurobi 完整问题参考成本：{reference['objective']:.6f}（{quality}）", flush=True)
    else:
        print(f"Gurobi 完整问题参考成本：未获得通过校验的参考方案。{reference.get('message', '')}", flush=True)
    if summary["reference_gap_percent"] is not None:
        print(f"主算法 UB 相对 Gurobi 参考成本的差距：{summary['reference_gap_percent']:.6f}%"
              "（(UB - 参考成本) / 参考成本 × 100%）", flush=True)
    else:
        print("主算法 UB 相对 Gurobi 参考成本的差距：无法计算（缺少通过校验的可行成本，或参考成本为零）。", flush=True)
    return int(result.status.endswith("FAILED") or result.validation is None
               or reference["validation"] is None)


# Windows 的独立认证进程需要保留此入口保护。
if __name__ == "__main__":
    raise SystemExit(main())
