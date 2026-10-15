"""修改顶部参数后，直接在 VS Code 中点击“运行 Python 文件”。"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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


# 修改这里后保存文件，再点击 VS Code 右上角的运行按钮。
CASE = "classic10"        # "classic10"：10机组；"rts24"：32机组
SEED = 17                # SA 随机种子
TIME_LIMIT_SECONDS = 120  # 主算法与完整问题参考求解各自的预算，单位：秒
TARGET_GAP_PERCENT = 0.1  # 认证间隙目标，单位：%；0.1 就是 0.1%
CHECK_ONLY = False       # True：只检查输入；False：实际求解

# 本入口关闭 LLM，使用 SA 与独立强化下界；子代理实验需另配调度器。
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def save(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_serializable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
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
    folders = {"classic10": "data", "rts24": "data/rts24/inputs"}
    if CASE not in folders:
        raise ValueError('CASE 请填写 "classic10" 或 "rts24"')
    folder = PROJECT / folders[CASE]
    options = dict(data_file=str(folder / "units.json"), scenario_file=str(folder / "stochastic.json"),
                   seed=SEED, time_limit=TIME_LIMIT_SECONDS,
                   tolerance=TARGET_GAP_PERCENT / 100, block_hours=3, max_outer=1000000,
                   max_inner=20, llm_enabled=False)
    checked = load_options(options)
    data, scenarios = load_data(checked.data_file, checked.scenario_file)
    dimensions = dict(units=data.n_units, hours=data.horizon, scenarios=len(scenarios))
    print(f"算例：{CASE}；{data.n_units}机组；{data.horizon}小时；{len(scenarios)}场景", flush=True)
    if CHECK_ONLY:
        print("输入检查通过；本次没有执行求解或检查授权。")
        return 0

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = RESULTS / f"manual-{CASE}-seed{SEED}-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    save(output / "options.json", options)
    print(f"结果目录：{output}", flush=True)
    print(f"随机种子：{SEED}；主算法预算：{TIME_LIMIT_SECONDS} 秒；"
          f"认证间隙目标：{TARGET_GAP_PERCENT}%；LLM：关闭", flush=True)
    print("指标说明：UB 是通过全天物理校验的可行方案成本；LB 是认证下界；"
          "认证间隙 = (UB - LB) / max(1, |UB|)。", flush=True)
    print("块内上下界用于分块搜索，不等于全天原问题的上下界。", flush=True)
    print("[1/2] 开始主算法：连续松弛初始化、分块 SA 搜索和独立强化下界认证。", flush=True)
    started = perf_counter()
    result = run_suc(options)
    save(output / "result.json", asdict(result))
    summary = dict(scope="EXPLORATORY_NO_INDEPENDENT_AUDIT", case=CASE, **dimensions,
                   seed=SEED, status=result.status, message=result.message,
                   upper=result.objective, lower=result.lower_bound,
                   gap_percent=100 * result.relative_gap, validation=result.validation,
                   solver_elapsed_seconds=result.elapsed_seconds,
                   wall_to_result_saved_seconds=perf_counter() - started,
                   sa_calls=result.diagnostics.get("sa_calls", 0), llm_calls=result.llm_calls,
                   note="入门实验：无增量检查点和独立审计；保存与完整问题参考求解在主算法预算之外。")
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

    print(f"\n[2/2] Gurobi 直接求解同一完整问题（保留全部整数变量、时段和场景），"
          f"另用最多 {TIME_LIMIT_SECONDS} 秒。", flush=True)
    print("参考求解仅用于比较，其方案和下界不回填主算法；日志见 reference-gurobi.log。", flush=True)
    reference_started = perf_counter()
    try:
        reference = solve_reference(data, scenarios, TIME_LIMIT_SECONDS, output / "reference-gurobi.log")
    except Exception as error:
        reference = dict(status="REFERENCE_FAILED", message=f"{type(error).__name__}: {error}",
                         objective=None, lower=None, gap_percent=None, validation=None,
                         elapsed_seconds=perf_counter() - reference_started)
    reference["time_limit_seconds"] = TIME_LIMIT_SECONDS
    save(output / "reference.json", reference)
    summary["reference"] = {key: value for key, value in reference.items()
                            if key not in ("u", "v", "w", "scenarios")}
    save(output / "summary.json", summary)
    print(f"结果已保存：{output}；完整参考方案见 reference.json。", flush=True)
    print(f"参考求解状态：{reference['status']}；耗时：{reference['elapsed_seconds']:.2f} 秒", flush=True)
    if reference["validation"] is not None:
        quality = "在求解器容差内最优" if reference["status"] == "OPTIMAL" else "可行参考方案，尚未证明最优"
        print(f"参考求解自身的最优间隙：{reference['gap_percent']:.6f}%", flush=True)
        print(f"Gurobi 完整问题参考成本：{reference['objective']:.6f}（{quality}）", flush=True)
    else:
        print(f"Gurobi 完整问题参考成本：未获得通过校验的参考方案。{reference.get('message', '')}", flush=True)
    return int(result.status.endswith("FAILED") or result.validation is None
               or reference["validation"] is None)


# Windows 的独立认证进程需要保留此入口保护。
if __name__ == "__main__":
    raise SystemExit(main())
