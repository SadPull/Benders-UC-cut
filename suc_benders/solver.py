"""Compute a reference, then run the time-block SA+Benders search."""
# 主循环只安排各阶段的执行顺序、预算与结束状态；候选处理在 search 中。
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from .model import model, partition
from .optimization import optimize
from .reference import solve_reference
from .search import BlockSearch, WholeDay, SABudgetExhausted
from .validation import validate

# 总截止时间提前这么多秒，留给收尾、最终校验和保存。
DEADLINE_MARGIN_SECONDS = 5.


# objective 是已验证方案费用；reference 是独立参考记录，不用于生成候选。
@dataclass
class Result:
    status: str = "RUNNING"
    message: str = ""
    objective: float = np.inf
    reference: dict = field(default_factory=dict)
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    w: np.ndarray | None = None
    scenarios: list = field(default_factory=list)
    validation: dict | None = None
    history: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.
    total_elapsed_seconds: float = 0.

    @property
    def reference_cost(self):
        cost = self.reference.get("objective")
        if cost is not None and np.isfinite(cost) and (self.reference.get("validation") or {}).get("feasible"):
            return float(cost)
        return None

    @property
    def reference_gap_percent(self):
        cost = self.reference_cost
        if cost is not None and cost != 0. and np.isfinite(self.objective) and (self.validation or {}).get("feasible"):
            return 100*(self.objective-cost)/abs(cost)
        return None

    def reference_target_reached(self, tolerance):
        cost = self.reference_cost
        return (self.reference.get("status") == "OPTIMAL" and cost is not None
                and np.isfinite(self.objective) and (self.validation or {}).get("feasible", False)
                and self.objective-cost <= 1e-5 + tolerance*abs(cost))


# 一次 solve 调用的共享状态。stage 只在进入新阶段时改写、此后保持不变，失败时报告为 <stage>_FAILED。
class _Run:
    def __init__(self, o, started, reference, total_started):
        self.o, self.started = o, started
        self.total_started = total_started
        self.deadline = started + o.time_limit - DEADLINE_MARGIN_SECONDS
        self.result, self.whole, self.searches, self.stage = Result(reference=reference), None, [], "INITIALIZATION"

    # WholeDay 尚未构造时使用本地 deadline，构造后转用当前工作配额调整后的 deadline。
    def remaining(self):
        return self.whole.remaining() if self.whole is not None else self.deadline - perf_counter()

    def require_time(self):
        if self.remaining() <= 0:
            raise TimeoutError

    def target_reached(self):
        if self.result.reference_target_reached(self.o.tolerance):
            self.result.status = "REFERENCE_TARGET_REACHED"
            return True
        return False


# 主流程：独立参考 → 初始化 → 分块搜索 → 整理结果。
# 保留阶段名定位失败，并在最终返回前独立复核全天方案。
def solve(d, scenarios, o, *, progress=None, started_at=None, reference_log_file=None, strategies=None):
    # started_at 是整体调用起点；搜索预算在参考结束后重新起计。
    total_started = perf_counter() if started_at is None else started_at
    if o.verbose:
        print(f"[参考求解] Gurobi 完整模型，独立预算 {o.reference_time_limit:g} 秒。", flush=True)
    reference = solve_reference(d, scenarios, o.reference_time_limit, reference_log_file)
    if o.verbose:
        print(f"[参考完成] 状态={reference['status']} 成本={reference['objective']}；开始独立搜索预算。", flush=True)
    run = _Run(o, perf_counter(), reference, total_started)
    result = run.result
    try:
        m = _initialize(run, d, scenarios, progress, strategies)
        try:
            _search(run, m)
        except SABudgetExhausted:
            run.whole.diagnostics['sa_budget_exhausted'] = True
            run.whole.diagnostics['search_stop_reason'] = 'SA_BUDGET'
            run.whole.emit('sa_budget_exhausted')
        if result.status != "REFERENCE_TARGET_REACHED":
            _finish_search(run)
    except Exception as error:
        # 只有截止时间信号归为 TIME_LIMIT；截止前后抛出的其他异常都是真实失败，保留阶段名。
        result.status = "TIME_LIMIT" if isinstance(error, TimeoutError) else f"{run.stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    _finalize(run, d, scenarios)
    return result


# 连续松弛只用于初始化；初始方案不使用参考解。
def _initialize(run, d, scenarios, progress, strategies=None):
    o, result = run.o, run.result
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    whole = run.whole = WholeDay(m, full, o, run.deadline, result, run.started, progress, strategies)
    phase = perf_counter()
    relaxation = optimize(m, run.remaining())
    whole.timings["initialization"] += perf_counter() - phase
    run.require_time()
    if relaxation.status != "OPTIMAL":
        raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
    whole.initialize(relaxation)
    if whole.best is None:
        raise RuntimeError("No validated initial schedule")
    whole.emit("initialization")
    run.target_reached()
    return m


# 按轮访问各提案块；全天割共享，块外方案每批固定。
def _search(run, m):
    o, whole = run.o, run.whole
    if not o.sa_enabled:
        whole.diagnostics['search_stop_reason'] = 'SA_DISABLED'
        return
    if run.result.status == "REFERENCE_TARGET_REACHED":
        return
    with whole.sa_work():
        run.stage = "SEARCH_SETUP"
        blocks, _ = partition(m, o.block_hours)
        run.searches = [BlockSearch(b, o, j + 1, whole) for j, b in enumerate(blocks)]
    for k in range(1, o.max_rounds + 1):
        if _run_round(run, k):
            return
    whole.diagnostics['search_stop_reason'] = 'MAX_ROUNDS'


# 一个完整遍历轮：每块各采样一批。返回 True 表示达到参考目标、不再继续轮询。
def _run_round(run, k):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    start = _round_start(run)
    record = dict(round=k, complete=False, visited_blocks=[], started_seconds=perf_counter()-run.started)
    whole.diagnostics["rounds"].append(record)
    for j, search in enumerate(searches):
        if run.target_reached():
            break
        run.require_time()
        _visit_block(run, j, search, k, record)
    if result.status == 'REFERENCE_TARGET_REACHED':
        record.update(ended_seconds=perf_counter()-run.started, stop_reason='REFERENCE_TARGET_REACHED')
        whole.emit('reference_target_reached')
        return True
    return _finish_round(run, k, start, record)


# 在一个 SA 工作配额作用域内完成单块提案、全天评估及割学习。
def _visit_block(run, j, search, k, record):
    o, result, whole = run.o, run.result, run.whole
    with whole.sa_work():
        run.stage = f"ROUND_{k}_BLOCK_{j + 1}"
        search.step(k)
        run.require_time()
        record["visited_blocks"].append(j + 1)
        if o.verbose:
            print(f"[分块进度] 轮次={k} 时间块={j + 1}/{len(run.searches)} "
                  f"全天可行成本(UB)={result.objective:.6f} "
                  f"参考成本={result.reference_cost}", flush=True)
        run.require_time()


# 完整遍历之后：更新 SA 轮计数，记录本轮迭代进度并刷新轮记录。
def _finish_round(run, k, start, record):
    result, whole = run.result, run.whole
    record.update(complete=True, sa_completed_seconds=perf_counter()-run.started,
                  incumbent_before=start.upper, incumbent_after_sa=result.objective)
    whole.diagnostics["complete_rounds"] += 1
    whole.diagnostics["iterations"].append(dict(iteration=len(whole.diagnostics["iterations"])+1, round=k,
                                                **_round_progress(run, start)))
    record.update(ended_seconds=perf_counter()-run.started, incumbent=result.objective)
    whole.emit("round")
    run.require_time()
    return run.target_reached()


# 一轮开始时的快照：UB、SA 调用数、评估数和两类割的累计数量。
def _round_start(run):
    optimality, feasibility = run.whole.cut_totals['optimality'], run.whole.cut_totals['feasibility']
    return NS(upper=run.result.objective, sa_calls=run.whole.diagnostics["sa_calls"],
              evaluations=len(run.whole.evaluated_commitments),
              optimality=optimality, feasibility=feasibility)


def _round_progress(run, start):
    optimality, feasibility = run.whole.cut_totals['optimality'], run.whole.cut_totals['feasibility']
    return dict(incumbent=run.result.objective, optimality_cuts_added=optimality-start.optimality,
                feasibility_cuts_added=feasibility-start.feasibility,
                new_candidates=len(run.whole.evaluated_commitments)-start.evaluations)


# 轮数、SA 配额用尽或 SA 关闭后结束搜索，不空等剩余预算。
def _finish_search(run):
    if not run.target_reached():
        run.require_time()
        run.result.status = "SEARCH_EXHAUSTED"


# 独立复核最终方案，参考差距由 Result 随当前方案计算。
def _finalize(run, d, scenarios):
    o, result, whole = run.o, run.result, run.whole
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if not result.validation['feasible']:
            result.status, result.message = "VALIDATION_FAILED", "Final schedule failed validation"
    if whole is not None:
        whole.diagnostics['search_statistics'] = whole.search_statistics()
    result.elapsed_seconds = perf_counter() - run.started
    result.total_elapsed_seconds = perf_counter() - run.total_started
    if whole is not None:
        whole.diagnostics['search_statistics']['new_feasible_per_second'] = (
            whole.diagnostics['search_statistics']['new_feasible']/max(result.elapsed_seconds, 1e-9))
        whole.timings["other"] = max(0., result.elapsed_seconds - sum(whole.timings.values()))
        result.diagnostics.update(evaluation_count=len(whole.diagnostics['evaluations']), cache_size=len(whole.cache),
                                  unique_evaluation_count=len(whole.evaluated_commitments),
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit))
        whole.emit("final")
    if o.verbose:
        print(f"[主算法结束] 状态={result.status} 可行成本(UB)={result.objective:.6f}, "
              f"参考成本={result.reference_cost}, "
              f"参考差距(%)={result.reference_gap_percent}, 搜索耗时={result.elapsed_seconds:.2f}秒", flush=True)
        failures = result.diagnostics.get("dispatch_failures", 0)
        if failures:
            print(f"[降级汇总] 调度数值失败 {failures} 次（只拒绝对应候选）。", flush=True)
