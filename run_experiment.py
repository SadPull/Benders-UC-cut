"""输入固定读取 data 中的 JSON；在 VS Code 中点击“运行 Python 文件”。"""
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from run_suc import run_suc, _serializable
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model
from suc_benders.optimization import optimize
from suc_benders.validation import unpack, validate


# 算例直接修改 data/units.json 和 data/stochastic.json；以下只设置运行参数。
SEED = 17                # SA 随机种子
TIME_LIMIT_SECONDS = 1800  # 主算法总预算，包含 Codex 启动、等待和回复处理，单位：秒
REFERENCE_TIME_LIMIT_SECONDS = 120  # 完整问题参考求解的独立预算，单位：秒
TARGET_GAP_PERCENT = 0.1  # 认证间隙目标，单位：%；0.1 就是 0.1%
CHECK_ONLY = False       # True：只检查输入；False：实际求解
LLM_ENABLED = True       # True：停滞时自动启动 Codex 独立任务；False：关闭 LLM
ENHANCED_SEARCH = True   # 全天可动机组的二元/三元联合 SA；False 仅运行分块搜索
BLOCK_SA_PERIOD = 3      # 联合搜索开启时，第 1、4、7…轮执行块 SA；调度与回填每轮保留
LLM_PATIENCE = 1         # 连续一个完整 SA 轮次改善不足 0.05% 时触发
LLM_MODEL = "gpt-6-astra"  # 使用 Codex 已登录账号，推理档位为 max
LLM_TIMEOUT_SECONDS = 1200  # 单次调用上限，仍受主算法剩余预算约束
LLM_RESERVE_SECONDS = 60  # 不用于等待 LLM，留给后续调度、校验和数值搜索

# Codex 提供上界候选，独立数值进程提供认证下界。
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def save(path, value):
    compressed = path.suffix == ".gz"
    with (gzip.open(path, "wt", encoding="utf-8") if compressed else path.open("w", encoding="utf-8")) as stream:
        json.dump(_serializable(value), stream, ensure_ascii=False,
                  indent=None if compressed else 2, allow_nan=False)
        stream.write("\n")


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
    output = RESULTS / f"manual-seed{SEED}-{stamp}-{uuid4().hex[:8]}"
    folder = PROJECT / "data"
    options = dict(data_file=str(folder / "units.json"), scenario_file=str(folder / "stochastic.json"),
                   seed=SEED, time_limit=TIME_LIMIT_SECONDS,
                   tolerance=TARGET_GAP_PERCENT / 100, block_hours=3, max_outer=1000000,
                   enhanced_search=ENHANCED_SEARCH, block_sa_period=BLOCK_SA_PERIOD,
                   max_inner=20, llm_enabled=LLM_ENABLED, llm_model=LLM_MODEL, llm_patience=LLM_PATIENCE,
                   llm_timeout=LLM_TIMEOUT_SECONDS, llm_reserve_seconds=LLM_RESERVE_SECONDS,
                   llm_codex_command="codex" if LLM_ENABLED else "",
                   llm_agent_directory=str(output / "llm_requests") if LLM_ENABLED else "")
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
    print(f"结果目录：{output}", flush=True)
    print(f"随机种子：{SEED}；主算法预算：{TIME_LIMIT_SECONDS} 秒；"
          f"认证间隙目标：{TARGET_GAP_PERCENT}%；LLM：{'开启' if LLM_ENABLED else '关闭'}", flush=True)
    if LLM_ENABLED:
        print(f"调用方式：Codex 独立任务；模型：{checked.llm_model}；推理档位：max；"
              f"连续 {checked.llm_patience} 个完整 SA 轮次改善不足时触发，"
              f"最多调用 {checked.llm_max_calls if checked.llm_max_calls is not None else '不限'} 次。", flush=True)
        print("复用 Codex 已有登录；请求和日志保存到 llm_requests；开启不代表立即调用。", flush=True)
        print(f"Codex 直接使用 HTTPS；单次最多等待 {checked.llm_timeout:g} 秒，"
              f"保留 {checked.llm_reserve_seconds:g} 秒给后续数值处理。", flush=True)
    print("指标说明：UB 是通过全天物理校验的可行方案成本；LB 是认证下界；"
          "认证间隙 = (UB - LB) / max(1, |UB|)。", flush=True)
    print("块内上下界用于分块搜索，不等于全天原问题的上下界。", flush=True)
    print("[1/2] 开始主算法：连续松弛初始化、分块 SA 搜索和独立强化下界认证。", flush=True)
    started = perf_counter()
    result = run_suc(options)
    save(output / "result.json.gz", result)
    summary = dict(scope="EXPLORATORY_NO_INDEPENDENT_AUDIT", **dimensions,
                   seed=SEED, status=result.status, message=result.message,
                   upper=result.objective, lower=result.lower_bound,
                   gap_percent=100 * result.relative_gap, reference_gap_percent=None,
                   validation=result.validation,
                   solver_elapsed_seconds=result.elapsed_seconds,
                   wall_to_result_saved_seconds=perf_counter() - started,
                   sa_calls=result.diagnostics.get("sa_calls", 0), llm_calls=result.llm_calls,
                   block_sa_calls=result.diagnostics.get('block_sa_calls', 0),
                   joint_sa_calls=result.diagnostics.get('joint_sa_calls', 0),
                   enhanced_search=ENHANCED_SEARCH,
                   block_sa_period=BLOCK_SA_PERIOD, llm_patience=LLM_PATIENCE,
                   search_statistics=result.diagnostics.get('search_statistics', {}),
                   note="入门实验：无增量检查点和独立审计；保存与完整问题参考求解在主算法预算之外。")
    save(output / "summary.json", summary)
    status_text = {"CONVERGED": "已达到认证间隙目标", "TIME_LIMIT": "达到时间上限",
                   "MAX_ITERATIONS": "达到迭代上限",
                   "PAUSED_LLM_FAILURE": "连续两次 LLM 调用失败，已停止主算法"}.get(result.status, "求解未正常完成")
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
    print(f"结果已保存：{output}；完整主结果见 result.json.gz，参考方案见 reference.json。", flush=True)
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
    return int(result.status.endswith("FAILED") or result.status == "PAUSED_LLM_FAILURE" or result.validation is None
               or reference["validation"] is None)


# Windows 的独立认证进程需要保留此入口保护。
if __name__ == "__main__":
    raise SystemExit(main())
