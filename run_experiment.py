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
TIME_LIMIT_SECONDS = 7200  # 搜索预算，包含记录开销，不含先行参考求解
REFERENCE_TIME_LIMIT_SECONDS = 120  # 完整问题参考求解的独立预算，单位：秒
TARGET_GAP_PERCENT = 0.1  # 相对参考成本差距目标，只有有效 OPTIMAL 参考允许提前停止
CHECK_ONLY = False       # True：只检查输入；False：实际求解
BLOCK_HOURS = 12         # 分块小时数；RCUC20 调参：3 h 块被长最小停机时间锁在块坐标局部最优
SA_SWEEPS = 16           # 每个温度对全部位的扫描次数；12 h 块下 16 优于默认 4
DATA_DIR = "data"        # 算例目录（相对项目根）；其他算例在 instances/<名称>

# 先行 Gurobi 参考只用于比较和停止，不注入候选搜索。
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def save(path, value):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False))
        stream.write("\n")


def freeze_experiment(output, options):
    paths = [PROJECT / name for name in ("run_experiment.py", "run_suc.py", "requirements.txt", "README.md")]
    paths += sorted((PROJECT / "suc_benders").glob("*.py"))
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
    effective = {f.name: getattr(options, f.name) for f in fields(options)}
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
                      ("numpy", "scipy", "gurobipy", "dwave-samplers", "dimod", "networkx")},
        files=hashes, effective_options_sha256=hashlib.sha256(
            (output / "effective-options.json").read_bytes()).hexdigest(),
        reference_usage="comparison_and_optimal_target_stop", reference_schedule_feedback=False))


# 当前最好可行方案的快照，不是可恢复的搜索状态。
def solution_snapshot(result, elapsed):
    values = {name: getattr(result, name) for name in
              ("status", "message", "objective", "u", "v", "w", "scenarios", "validation")}
    return dict(values, format="SOLUTION_SNAPSHOT_NOT_CHECKPOINT", reference_cost=result.reference_cost,
                reference_status=result.reference.get("status"), reference_gap_percent=result.reference_gap_percent,
                sa_backend=result.diagnostics.get("sa_backend"),
                sa_initialization=result.diagnostics.get("sa_initialization"),
                constraint_policy=result.diagnostics.get("constraint_policy"),
                cut_selector=result.diagnostics.get("cut_selector"),
                elapsed_seconds=elapsed,
                model_fingerprint=result.diagnostics.get("model_fingerprint"),
                sa_calls=result.diagnostics.get("sa_calls", 0))


class ExperimentProgress:
    def __init__(self, output):
        self.output = output
        self.last_upper = np.inf
        self.recording_seconds = 0.

    def __call__(self, whole, event):
        result = whole.result
        phase = perf_counter()
        elapsed = phase - whole.started
        improved = result.objective < self.last_upper
        if event in ("before_sampling", "sampling") or (event == "evaluation" and not improved):
            return
        self.last_upper = result.objective
        row = dict(event=event, elapsed_seconds=elapsed, upper=result.objective,
                   reference_cost=result.reference_cost, reference_status=result.reference.get("status"),
                   reference_gap_percent=result.reference_gap_percent,
                   sa_calls=whole.diagnostics["sa_calls"], status=result.status)
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        if improved or event in ("initialization", "final"):
            temporary = self.output / "latest-solution.tmp.json"
            save(temporary, solution_snapshot(result, elapsed))
            temporary.replace(self.output / "latest-solution.json")
        if event == "initialization":
            save(self.output / "reference.json", result.reference)
        self.recording_seconds += perf_counter() - phase


# 由文件顶部常量组装求解参数。
def build_options(folder):
    return dict(data_file=str(folder / "units.json"), scenario_file=str(folder / "stochastic.json"),
                seed=SEED, time_limit=TIME_LIMIT_SECONDS, reference_time_limit=REFERENCE_TIME_LIMIT_SECONDS,
                tolerance=TARGET_GAP_PERCENT / 100, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS, max_rounds=1000000)


def print_run_settings():
    print(f"随机种子：{SEED}；主算法预算：{TIME_LIMIT_SECONDS} 秒；"
          f"参考成本差距目标：{TARGET_GAP_PERCENT}%；分块：{BLOCK_HOURS} 小时；SA sweeps：{SA_SWEEPS}", flush=True)
    print("指标说明：UB 是通过全天物理校验的可行方案成本；"
          "参考差距 = (UB - 参考成本) / |参考成本| × 100%，允许为负；参考成本为零时不计算百分比。", flush=True)
    print("先行参考预算独立；只有通过校验的 OPTIMAL 参考允许按目标提前停止。", flush=True)


# 汇总搜索与独立参考的状态、差距、计时和调用统计。
def run_summary(result, progress, dimensions, solver_wall, final_save_seconds, started):
    return dict(**dimensions, seed=SEED, status=result.status, message=result.message,
                sa_backend=result.diagnostics.get("sa_backend"),
                sa_initialization=result.diagnostics.get("sa_initialization"),
                constraint_policy=result.diagnostics.get("constraint_policy"),
                cut_selector=result.diagnostics.get("cut_selector"),
                upper=result.objective, reference_gap_percent=result.reference_gap_percent,
                reference={key: value for key, value in result.reference.items()
                           if key not in ("u", "v", "w", "scenarios")},
                validation=result.validation,
                solver_elapsed_seconds=result.elapsed_seconds, solver_wall_seconds=solver_wall,
                reference_elapsed_seconds=result.reference["elapsed_seconds"],
                total_elapsed_seconds=result.total_elapsed_seconds,
                recording_seconds=progress.recording_seconds, final_save_seconds=final_save_seconds,
                time_limit_overshoot_seconds=result.diagnostics.get("time_limit_overshoot", 0.),
                wall_to_result_saved_seconds=perf_counter() - started,
                sa_calls=result.diagnostics.get("sa_calls", 0),
                dynamic_constraints_added=result.diagnostics.get("dynamic_constraints_added", 0),
                data_dir=DATA_DIR, block_hours=BLOCK_HOURS, sa_sweeps=SA_SWEEPS,
                dispatch_failures=result.diagnostics.get('dispatch_failures', 0),
                complete_rounds=result.diagnostics.get('complete_rounds', 0),
                search_stop_reason=result.diagnostics.get('search_stop_reason'),
                sa_budget_exhausted=bool(result.diagnostics.get('sa_budget_exhausted')),
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
          f"SA 调用：{summary['sa_calls']} 次", flush=True)
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
    options = build_options(PROJECT / DATA_DIR)
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
    print_run_settings()
    print("运行顺序：先计算 Gurobi 参考，再初始化并执行分块 SA+Benders 搜索。", flush=True)
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
    compact = solution_snapshot(result, result.elapsed_seconds)
    compact.update(timings=result.diagnostics.get("timings", {}),
                   search_statistics=result.diagnostics.get("search_statistics", {}),
                   reference={key: value for key, value in result.reference.items()
                              if key not in ("u", "v", "w", "scenarios")},
                   total_elapsed_seconds=result.total_elapsed_seconds)
    save(output / "result.json", compact)
    save(output / "reference.json", result.reference)
    final_save_seconds = perf_counter() - final_save_started
    summary = run_summary(result, progress, dimensions, solver_wall, final_save_seconds, started)
    save(output / "summary.json", summary)
    print_result(result, summary)
    print(f"结果已保存：{output}；精简主结果见 result.json，参考方案见 reference.json。", flush=True)
    return int(result.status.endswith("FAILED") or not (result.validation or {}).get("feasible"))


# 入口保护：作为模块导入时不启动求解。
if __name__ == "__main__":
    raise SystemExit(main())
