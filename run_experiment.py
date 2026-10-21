"""一键运行入口：参数在文件顶部设置，算例目录由 DATA_DIR 选择；在 VS Code 中点击“运行 Python 文件”。"""
from datetime import datetime, timedelta, timezone
from dataclasses import fields
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

from run_suc import run_suc
from suc_benders.anneal import SA_BACKEND, SA_INITIALIZATION
from suc_benders.config import load_options
from suc_benders.data import jsonable, load_data


# 默认算例为 data/；切换算例改 DATA_DIR。以下只设置运行参数。
SEED = 17                # SA 随机种子
TIME_LIMIT_SECONDS = 7200  # 搜索预算，包含 LLM 和记录开销，不含先行参考求解
REFERENCE_TIME_LIMIT_SECONDS = 120  # 完整问题参考求解的独立预算，单位：秒
TARGET_GAP_PERCENT = 0.1  # 相对参考成本差距目标，只有有效 OPTIMAL 参考允许提前停止
CHECK_ONLY = False       # True：只检查输入；False：实际求解
LLM_ENABLED = True       # True：停滞时自动请求 LLM 建议（通道见 LLM_TRANSPORT）；False：关闭 LLM
LLM_PATIENCE = 1         # 连续一个完整 SA 轮次改善不足 0.05% 时触发
LLM_FIRST_CALL_AT_START = True  # True：初始化后立即发首个请求，不等待 SA 停滞
LLM_MODEL = "gpt-6-astra"  # 使用 Codex 已登录账号，推理档位为 max
LLM_TRANSPORT = "codex"  # "codex"：Codex 独立任务；"http"：流式 Responses，URL/密钥取自 config.local.json
LLM_TIMEOUT_SECONDS = 1800  # 剩余时间不足完整请求窗口加预留时，不启动新请求
LLM_RESERVE_SECONDS = 60  # 不用于等待 LLM，留给后续调度、校验和数值搜索
LLM_MAX_CALLS = None     # 全程调用上限；None：在预算内按停滞规则继续，保留累计调用计数
LLM_ESCALATION_MODEL = ""  # 升级层模型（经 Codex CLI，隔离用户配置、不续接）；"" 关闭。需 LLM_TRANSPORT="http"
LLM_ESCALATION_GAIN = 0.005  # 常规层一次调用降幅低于此比例（0.005 即 0.5%）或调用失败时，并行发起一次升级请求
LLM_ESCALATION_MIN_SECONDS = 700  # 剩余预算扣除预留后不足此秒数时不发起升级请求
LLM_ESCALATION_TIMEOUT_SECONDS = 1800  # 升级层单次上限；实际还不超过剩余预算扣除预留
BLOCK_HOURS = 12         # 分块小时数；RCUC20 调参：3 h 块被长最小停机时间锁在块坐标局部最优
SA_SWEEPS = 16           # 每个温度对全部位的扫描次数；12 h 块下 16 优于默认 4
DATA_DIR = "data"        # 算例目录（相对项目根）；其他算例在 instances/<名称>

# Codex 提供候选；先行 Gurobi 参考只用于比较和停止，不注入候选搜索。
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def save(path, value):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False))
        stream.write("\n")


def freeze_experiment(output, options):
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
        reference_usage="comparison_and_optimal_target_stop", reference_schedule_feedback=False,
        reference_in_llm_prompt=False))


# 当前最好可行方案的快照，不是可恢复的搜索状态。
def solution_snapshot(result, elapsed, session_id=None):
    values = {name: getattr(result, name) for name in
              ("status", "message", "objective", "u", "v", "w", "scenarios", "validation", "llm_calls")}
    return dict(values, format="SOLUTION_SNAPSHOT_NOT_CHECKPOINT", reference_cost=result.reference_cost,
                reference_status=result.reference.get("status"), reference_gap_percent=result.reference_gap_percent,
                sa_backend=result.diagnostics.get("sa_backend"),
                sa_initialization=result.diagnostics.get("sa_initialization"),
                elapsed_seconds=elapsed, session_id=session_id,
                model_fingerprint=result.diagnostics.get("model_fingerprint"),
                sa_calls=result.diagnostics.get("sa_calls", 0))


# 精简一次 LLM 调用的回执：保留请求基准 base_u 与模型原样改动 raw_changes（二者可还原完整方案），
# 逐候选写出格式剔除原因、评估结果、调度状态与费用，以及移植结果；不保存可由此重建的整张矩阵。
# accepted 是评估后被接受的候选数，format_valid/invalid 是解析阶段的合格/剔除数。
def llm_record(row):
    record = {key: row.get(key) for key in (
        "call_id", "status", "thread_id", "requested_session_id", "request_directory", "background_seconds",
        "request_upper", "reference_cost", "upper_before_probes", "upper_after_probes", "prompt_mode", "usage", "message",
        "accepted", "format_valid", "invalid", "base_u", "response_model", "response_output",
        "tier", "triggered_by", "transport", "requested_model", "deadline_seconds", "improved",
        "codex_home_cleanup")}
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
    def __init__(self, output):
        self.output = output
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
        row = dict(event=event, elapsed_seconds=elapsed, upper=result.objective,
                   reference_cost=result.reference_cost, reference_status=result.reference.get("status"),
                   reference_gap_percent=result.reference_gap_percent,
                   sa_calls=whole.diagnostics["sa_calls"], llm_calls=result.llm_calls,
                   status=result.status, session_id=self.session_id)
        # 两层并行时，事件对应的调用不一定是 llm_history 的最后一行，取顾问记下的事件行。
        llm_row = whole.advice.event_row if event.startswith("llm") else None
        if llm_row is not None:
            row.update(llm_call_id=llm_row.get("call_id"), llm_tier=llm_row.get("tier"),
                       llm_status=llm_row.get("status"))
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        if improved or event in ("initialization", "final"):
            temporary = self.output / "latest-solution.tmp.json"
            save(temporary, solution_snapshot(result, elapsed, self.session_id))
            temporary.replace(self.output / "latest-solution.json")
        if event == "initialization":
            save(self.output / "reference.json", result.reference)
        self.recording_seconds += perf_counter() - phase


# 由文件顶部常量组装求解参数；LLM 通道与升级层的组合不合法时直接报错，不创建结果目录。
def build_options(output, folder):
    if LLM_TRANSPORT not in ("codex", "http"):
        raise ValueError("LLM_TRANSPORT must be 'codex' or 'http'")
    codex = LLM_ENABLED and LLM_TRANSPORT == "codex"
    escalation = LLM_ENABLED and bool(LLM_ESCALATION_MODEL)
    if escalation and LLM_TRANSPORT != "http":
        raise ValueError("LLM escalation requires LLM_TRANSPORT='http' for the regular tier")
    options = dict(data_file=str(folder / "units.json"), scenario_file=str(folder / "stochastic.json"),
                   seed=SEED, time_limit=TIME_LIMIT_SECONDS, reference_time_limit=REFERENCE_TIME_LIMIT_SECONDS,
                   tolerance=TARGET_GAP_PERCENT / 100, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS, max_rounds=1000000,
                   llm_enabled=LLM_ENABLED, llm_model=LLM_MODEL, llm_patience=LLM_PATIENCE,
                   llm_max_calls=LLM_MAX_CALLS, llm_first_call_at_start=LLM_FIRST_CALL_AT_START,
                   llm_timeout=LLM_TIMEOUT_SECONDS, llm_reserve_seconds=LLM_RESERVE_SECONDS,
                   llm_codex_command="codex" if codex else "",
                   llm_agent_directory=str(output / "llm_requests") if codex else "",
                   llm_escalation_model=LLM_ESCALATION_MODEL if escalation else "",
                   llm_escalation_command="codex" if escalation else "",
                   llm_escalation_directory=str(output / "llm_requests") if escalation else "",
                   llm_escalation_gain=LLM_ESCALATION_GAIN, llm_escalation_min_seconds=LLM_ESCALATION_MIN_SECONDS,
                   llm_escalation_timeout=LLM_ESCALATION_TIMEOUT_SECONDS)
    return options, codex, escalation


def print_run_settings(checked, codex, escalation):
    print(f"随机种子：{SEED}；主算法预算：{TIME_LIMIT_SECONDS} 秒；"
          f"参考成本差距目标：{TARGET_GAP_PERCENT}%；LLM：{'开启' if LLM_ENABLED else '关闭'}", flush=True)
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
        if escalation:
            print(f"升级层：{checked.llm_escalation_model}（Codex 独立任务，推理 max），与常规层并行；"
                  f"常规层一次调用降幅低于 {100*checked.llm_escalation_gain:g}% 或失败时发起，"
                  f"剩余预算扣除预留不足 {checked.llm_escalation_min_seconds:g} 秒时不发起，"
                  f"单次最多 {checked.llm_escalation_timeout:g} 秒。", flush=True)
            print("升级层每次发送完整提示、不续接会话，在只含登录信息与过滤配置的临时 CODEX_HOME 中运行，"
                  "调用结束即删除；请求与日志保存到 llm_requests。", flush=True)
    print("指标说明：UB 是通过全天物理校验的可行方案成本；"
          "参考差距 = (UB - 参考成本) / |参考成本| × 100%，允许为负；参考成本为零时不计算百分比。", flush=True)
    print("先行参考预算独立；只有通过校验的 OPTIMAL 参考允许按目标提前停止。", flush=True)


# 汇总搜索与独立参考的状态、差距、计时和调用统计。
def run_summary(result, checked, progress, dimensions, solver_wall, final_save_seconds, started):
    return dict(**dimensions, seed=SEED, status=result.status, message=result.message,
                sa_backend=result.diagnostics.get("sa_backend"),
                sa_initialization=result.diagnostics.get("sa_initialization"),
                upper=result.objective, reference_gap_percent=result.reference_gap_percent,
                reference={key: value for key, value in result.reference.items()
                           if key not in ("u", "v", "w", "scenarios")},
                validation=result.validation,
                solver_elapsed_seconds=result.elapsed_seconds, solver_wall_seconds=solver_wall,
                reference_elapsed_seconds=result.reference["elapsed_seconds"],
                total_elapsed_seconds=result.total_elapsed_seconds,
                recording_seconds=progress.recording_seconds, final_save_seconds=final_save_seconds,
                session_id=progress.session_id,
                time_limit_overshoot_seconds=result.diagnostics.get("time_limit_overshoot", 0.),
                wall_to_result_saved_seconds=perf_counter() - started,
                sa_calls=result.diagnostics.get("sa_calls", 0), llm_calls=result.llm_calls,
                llm_patience=LLM_PATIENCE, llm_first_call_at_start=LLM_FIRST_CALL_AT_START,
                llm_disabled=result.diagnostics.get('llm_disabled'),
                llm_escalation_model=checked.llm_escalation_model or None,
                llm_escalation_calls=sum(row.get("tier") == "escalation" for row in result.llm_history),
                llm_escalation_improved=sum(bool(row.get("improved")) for row in result.llm_history
                                            if row.get("tier") == "escalation"),
                llm_escalation_disabled=result.diagnostics.get('llm_escalation_disabled'),
                data_dir=DATA_DIR, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS,
                dispatch_failures=result.diagnostics.get('dispatch_failures', 0),
                search_statistics=result.diagnostics.get('search_statistics', {}),
                note="轻量留证；latest-solution.json 仅保存方案，不是可恢复的搜索状态。")


def print_result(result, summary):
    status_text = {"REFERENCE_TARGET_REACHED": "已达到参考成本差距目标", "SEARCH_EXHAUSTED": "搜索轮次或配额已用尽",
                   "TIME_LIMIT": "达到搜索时间上限"}.get(result.status, "求解未正常完成")
    print(f"\n主算法状态：{status_text}（{result.status}）", flush=True)
    if result.message:
        print(f"状态详情：{result.message}", flush=True)
    if result.validation is not None:
        print(f"主算法可行成本（UB）：{result.objective:.6f}", flush=True)
    else:
        print("主算法尚未获得通过校验的全天可行方案。", flush=True)
    print(f"主算法耗时：{result.elapsed_seconds:.2f} 秒；"
          f"SA 调用：{summary['sa_calls']} 次；LLM 调用：{result.llm_calls} 次", flush=True)
    reference = result.reference
    print(f"参考求解状态：{reference['status']}；耗时：{reference['elapsed_seconds']:.2f} 秒", flush=True)
    if result.reference_cost is not None:
        quality = "在求解器容差内最优" if reference["status"] == "OPTIMAL" else "可行参考方案，尚未证明最优"
        if reference.get("gurobi_gap_percent") is not None:
            print(f"参考求解自身的最优间隙：{reference['gurobi_gap_percent']:.6f}%", flush=True)
        print(f"Gurobi 完整问题参考成本：{reference['objective']:.6f}（{quality}）", flush=True)
    else:
        print(f"Gurobi 完整问题参考成本：未获得通过校验的参考方案。{reference.get('message', '')}", flush=True)
    if summary["reference_gap_percent"] is not None:
        print(f"主算法 UB 相对 Gurobi 参考成本的差距：{summary['reference_gap_percent']:.6f}%"
              "（(UB - 参考成本) / |参考成本| × 100%）", flush=True)
    else:
        print("主算法 UB 相对 Gurobi 参考成本的差距：无法计算（缺少通过校验的可行成本，或参考成本为零）。", flush=True)
    print(f"参考与搜索总耗时：{result.total_elapsed_seconds:.2f} 秒", flush=True)


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = RESULTS / f"manual-seed{SEED}-{stamp}-{uuid4().hex[:8]}"
    options, codex, escalation = build_options(output, PROJECT / DATA_DIR)
    checked = load_options(options)
    data, scenarios = load_data(checked.data_file, checked.scenario_file)
    dimensions = dict(units=data.n_units, hours=data.horizon, scenarios=len(scenarios))
    print(f"机组输入：{checked.data_file}", flush=True)
    print(f"场景输入：{checked.scenario_file}", flush=True)
    print(f"输入规模：{data.n_units}机组；{data.horizon}小时；{len(scenarios)}场景", flush=True)
    if CHECK_ONLY:
        print("输入检查通过；本次没有执行求解或检查授权。")
        return 0

    output.mkdir(parents=True, exist_ok=False)
    save(output / "options.json", options)
    freeze_experiment(output, checked)
    progress = ExperimentProgress(output)
    print(f"结果目录：{output}", flush=True)
    print_run_settings(checked, codex, escalation)
    print("运行顺序：先计算 Gurobi 参考，再初始化并执行分块 SA/LLM 搜索。", flush=True)
    started = perf_counter()
    try:
        result = run_suc(vars(checked), progress=progress, started_at=started,
                         reference_log_file=output / "reference-gurobi.log")
    except Exception as error:
        save(output / "failure.json", dict(status="INCOMPLETE", error_type=type(error).__name__,
                                          elapsed_seconds=perf_counter() - started))
        raise
    solver_wall = perf_counter() - started - result.reference["elapsed_seconds"]
    final_save_started = perf_counter()
    compact = solution_snapshot(result, result.elapsed_seconds, progress.session_id)
    compact.update(timings=result.diagnostics.get("timings", {}),
                   search_statistics=result.diagnostics.get("search_statistics", {}),
                   reference={key: value for key, value in result.reference.items()
                              if key not in ("u", "v", "w", "scenarios")},
                   total_elapsed_seconds=result.total_elapsed_seconds,
                   llm_history=[llm_record(row) for row in result.llm_history])
    save(output / "result.json", compact)
    save(output / "reference.json", result.reference)
    final_save_seconds = perf_counter() - final_save_started
    summary = run_summary(result, checked, progress, dimensions, solver_wall, final_save_seconds, started)
    save(output / "summary.json", summary)
    print_result(result, summary)
    print(f"结果已保存：{output}；精简主结果见 result.json，参考方案见 reference.json。", flush=True)
    return int(result.status.endswith("FAILED") or not (result.validation or {}).get("feasible"))


# Windows 的 LLM 工作进程需要保留此入口保护。
if __name__ == "__main__":
    raise SystemExit(main())
