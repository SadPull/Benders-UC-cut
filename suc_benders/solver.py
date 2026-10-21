"""Compute a reference, then coordinate time-block search and upper advice."""
# 主循环只安排各阶段的执行顺序、预算与结束状态；候选处理在 search 中。
from dataclasses import dataclass, field
from time import perf_counter, sleep
from types import SimpleNamespace as NS

import numpy as np

from .benders import coordinate
from .model import model, partition
from .optimization import optimize
from .reference import solve_reference
from .search import BlockSearch, WholeDay, SABudgetExhausted, closed
from .validation import validate

# 总截止时间提前这么多秒，留给收尾、最终校验和保存。
DEADLINE_MARGIN_SECONDS = 5.


# objective 是已验证方案费用；reference 是独立参考记录，不用于生成候选。
# coordination_upper 仅属于内部协调对偶模型。
@dataclass
class Result:
    status: str = "RUNNING"
    message: str = ""
    objective: float = np.inf
    reference: dict = field(default_factory=dict)
    coordination_upper: float = np.inf
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    w: np.ndarray | None = None
    scenarios: list = field(default_factory=list)
    validation: dict | None = None
    history: list = field(default_factory=list)
    outer_history: list = field(default_factory=list)
    llm_calls: int = 0
    llm_history: list = field(default_factory=list)
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


# 主流程：独立参考 → 初始化 → 分块搜索/建议 → 协调更新价格 → 整理结果。
# 保留阶段名定位失败，并在最终返回前独立复核全天方案。
def solve(d, scenarios, o, *, progress=None, started_at=None, reference_log_file=None):
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
        m, relaxation = _initialize(run, d, scenarios, progress)
        try:
            _search(run, m, relaxation)
        except SABudgetExhausted:
            run.whole.diagnostics['sa_budget_exhausted'] = True
            run.whole.emit('sa_budget_exhausted')
        if result.status != "REFERENCE_TARGET_REACHED":
            _finish_advice(run)
    except Exception as error:
        result.status = ("TIME_LIMIT" if run.remaining() <= 0 or isinstance(error, TimeoutError)
                         else f"{run.stage}_FAILED")
        result.message = f"{type(error).__name__}: {error}"
    finally:
        _shutdown(run)
    _finalize(run, d, scenarios)
    return result


# 连续松弛只用于初始化与内部协调；初始方案不使用参考解。
def _initialize(run, d, scenarios, progress):
    o, result = run.o, run.result
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    whole = run.whole = WholeDay(m, full, o, run.deadline, result, run.started, progress)
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
    if not run.target_reached():
        whole.advice.kickoff()
    return m, relaxation


# 外层循环：每个外层用当前乘子重建各块，做至多 max_inner 个完整轮，再协调更新跨块乘子。
def _search(run, m, relaxation):
    o, result, whole = run.o, run.result, run.whole
    prices = relaxation.prices
    with whole.sa_work():
        blocks, cross = partition(m, o.block_hours, prices) if o.sa_enabled else ([], np.array([], int))
    planes, previous = [[] for _ in blocks], [[] for _ in blocks]
    for outer in (range(1, o.max_outer + 1) if o.sa_enabled and result.status != "REFERENCE_TARGET_REACHED" else []):
        run.require_time()
        with whole.sa_work():
            # 新乘子对应新的块费用与局部割环境，历史二进制种子必须重新调度。
            blocks, _ = partition(m, o.block_hours, prices)
            run.searches = [BlockSearch(b, o, outer, j + 1, whole, planes[j]) for j, b in enumerate(blocks)]
        # LLM 批次可以跨越协调：新乘子的块建好后先推进在途批次。
        if whole.advice.state["phase"] == "llm":
            run.stage = "ADVISOR_CARRYOVER"
            whole.advice.advance(run.searches)
            if run.target_reached():
                break
        rounds = _run_rounds(run, outer, previous)
        if result.status == "REFERENCE_TARGET_REACHED":
            break
        whole.advice.advance(run.searches)
        if run.target_reached():
            break
        prices, previous = _coordinate(run, m, outer, rounds, prices, planes, cross, relaxation.lower)


# 运行一个外层内的完整轮，返回最后一轮的序号（记入协调记录的 complete_rounds）。
def _run_rounds(run, outer, previous):
    for inner in range(1, run.o.max_inner + 1):
        if _run_round(run, outer, inner, previous):
            break
    return inner


# 一个完整遍历轮：尚未闭合的块各最多采样一批，已闭合块跳过采样。返回 True 表示本外层不再继续轮询。
def _run_round(run, outer, inner, previous):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    start = _round_start(run)
    record = dict(outer=outer, round=inner, global_round=whole.diagnostics['complete_rounds']+1,
                  complete=False, visited_blocks=[], skipped_closed_blocks=[],
                  started_seconds=perf_counter()-run.started)
    whole.diagnostics["rounds"].append(record)
    for j, search in enumerate(searches):
        whole.advice.advance(searches)
        if run.target_reached():
            break
        run.require_time()
        if closed(search.lower, search.upper, o.tolerance):
            record["skipped_closed_blocks"].append(j + 1)
        else:
            _visit_block(run, j, search, outer, inner, previous, record)
        whole.advice.advance(searches)
    if result.status == 'REFERENCE_TARGET_REACHED':
        record.update(ended_seconds=perf_counter()-run.started, stop_reason='REFERENCE_TARGET_REACHED')
        whole.emit('reference_target_reached')
        return True
    return _finish_round(run, outer, inner, start, record)


# 在一个 SA 工作配额作用域内完成单块访问：块搜索，再把块内最好方案拼回全天做 pool 候选。
def _visit_block(run, j, search, outer, inner, previous, record):
    o, result, whole = run.o, run.result, run.whole
    with whole.sa_work():
        b = search.b
        run.stage = f"BLOCK_{outer}_{j + 1}_ROUND_{inner}"
        seeds = [whole.best.point[b.cols[b.binary]], *(previous[j] if inner == 1 else []),
                 *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
        search.step(seeds, inner)
        run.require_time()
        record["visited_blocks"].append(j + 1)
        if search.best is not None:
            raw = result.u.copy(); raw[:, b.times] = search.best.point[b.u]
            proposal = {"proposal_id": f"pool:{outer}:{inner}:{j+1}", "outer": outer, "round": inner, "block": j + 1}
            x = whole.prepare(raw, "pool", proposal)
            whole.accept(x, "pool", proposal)
        if o.verbose:
            print(f"[分块进度] 外层={outer} 内层轮次={inner} 时间块={j + 1}/{len(run.searches)} "
                  f"块内[下界, 上界]=[{search.lower:.6f}, {search.upper:.6f}] "
                  f"全天可行成本(UB)={result.objective:.6f} "
                  f"参考成本={result.reference_cost}", flush=True)
        run.require_time()


# 完整遍历之后：更新 SA 轮计数、处理上界建议，最后刷新轮记录。
# 迭代记录须在 after_round 之前写入（它会进入 LLM 提示）；建议阶段可能新增割和评估，因此轮末再统计一次。
def _finish_round(run, outer, inner, start, record):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    record.update(complete=True, sa_completed_seconds=perf_counter()-run.started,
                  incumbent_before=start.upper, incumbent_after_sa=result.objective)
    whole.diagnostics["complete_rounds"] += 1
    iteration = dict(iteration=len(whole.diagnostics["iterations"])+1, outer=outer, round=inner,
                     **_round_progress(run, start), llm_failed=False)
    whole.diagnostics["iterations"].append(iteration)
    all_closed = all(closed(x.lower, x.upper, o.tolerance) for x in searches)
    run.stage = "ADVISOR"
    whole.advice.after_round(start.upper, start.sa_calls, searches, outer, inner,
                             not (outer == o.max_outer and inner == o.max_inner))
    iteration.update(**_round_progress(run, start),
                     llm_failed=any(row["status"] not in ("OK", "RUNNING", "CANCELLED")
                                    for row in result.llm_history[start.llm_calls:]))
    record.update(ended_seconds=perf_counter()-run.started, incumbent=result.objective,
                  llm_calls=result.llm_calls-start.llm_calls)
    whole.emit("round")
    for search in searches:
        search.summarize()
    run.require_time()
    return run.target_reached() or all_closed


# 一轮开始时的快照：UB、SA 调用数、评估数、LLM 调用数和两类割数量。
def _round_start(run):
    optimality, feasibility = _cut_counts(run.searches)
    return NS(upper=run.result.objective, sa_calls=run.whole.diagnostics["sa_calls"],
              evaluations=len(run.whole.diagnostics["evaluations"]), llm_calls=run.result.llm_calls,
              optimality=optimality, feasibility=feasibility)


def _round_progress(run, start):
    optimality, feasibility = _cut_counts(run.searches)
    return dict(incumbent=run.result.objective, optimality_cuts_added=optimality-start.optimality,
                feasibility_cuts_added=feasibility-start.feasibility,
                new_candidates=len(run.whole.diagnostics["evaluations"])-start.evaluations)


def _cut_counts(searches):
    return (sum(c.optimality for search in searches for c in search.cuts),
            sum(not c.optimality for search in searches for c in search.cuts))


# 用各块可行平面协调跨块乘子，记录协调对偶上界与下界；返回新乘子和下一外层的重调度种子。
def _coordinate(run, m, outer, rounds, prices, planes, cross, relaxation_lower):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    with whole.sa_work():
        previous = [search.seeds for search in searches]
        run.stage = "COORDINATION"
        phase = perf_counter()
        values, upper, info = coordinate(planes, m.eq[cross], result.objective, run.remaining(), prices[cross])
        whole.timings["coordination"] += perf_counter() - phase
        run.require_time()
        result.coordination_upper = min(result.coordination_upper, upper)
        result.outer_history.append({"outer": outer, "reference_cost": result.reference_cost, "coordination_upper": upper,
                                     "coordination": info, "complete_rounds": rounds,
                                     "unconverged_blocks": [x.index for x in searches
                                                            if not closed(x.lower, x.upper, o.tolerance)]})
        whole.diagnostics["coordination_count"] += 1
        whole.emit("coordination")
        dual_lower = max(relaxation_lower, float(sum(search.lower for search in searches)))
        result.outer_history[-1]["dual_lower"] = dual_lower
        result.outer_history[-1]["dual_converged"] = closed(dual_lower, upper, o.tolerance, dual_lower)
        prices = np.zeros(len(m.rhs)); prices[cross] = values
    return prices, previous


# SA 用尽后只收尾已有批次及其升级请求，没有工作时不空等预算。
def _finish_advice(run):
    run.stage = "ADVISOR"
    advice = run.whole.advice
    while not run.target_reached():
        run.require_time()
        if (advice.state['batch'] is None and advice.regular.worker is None
                and advice.escalation.worker is None and not advice.escalation_ready()):
            run.result.status = "SEARCH_EXHAUSTED"
            return
        advice.advance(run.searches)
        sleep(min(.05, max(0., run.remaining())))


# 取消在途 LLM 请求。
def _shutdown(run):
    whole = run.whole
    if whole is None:
        return
    whole.advice.close()


# 独立复核最终方案，参考差距由 Result 随当前方案计算。
def _finalize(run, d, scenarios):
    o, result, whole = run.o, run.result, run.whole
    for search in run.searches:
        search.summarize()
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
        result.diagnostics.update(evaluation_count=len(whole.cache), llm_call_count=result.llm_calls,
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit))
        whole.emit("final")
    if o.verbose:
        print(f"[主算法结束] 状态={result.status} 可行成本(UB)={result.objective:.6f}, "
              f"参考成本={result.reference_cost}, "
              f"参考差距(%)={result.reference_gap_percent}, 搜索耗时={result.elapsed_seconds:.2f}秒", flush=True)
