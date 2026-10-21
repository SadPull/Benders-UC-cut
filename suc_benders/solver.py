"""Compute a reference, maximise the cross-block Lagrangian, then run time-block search and upper advice."""
# 主循环只安排各阶段的执行顺序、预算与结束状态；候选处理在 search 中。
from dataclasses import dataclass, field
from time import perf_counter, sleep
from types import SimpleNamespace as NS

import numpy as np

from . import lagrangian
from .model import model, partition, priced, relaxed_rows, search_multipliers
from .optimization import optimize
from .reference import solve_reference
from .search import BlockSearch, WholeDay, SABudgetExhausted
from .validation import validate

# 总截止时间提前这么多秒，留给收尾、最终校验和保存。
DEADLINE_MARGIN_SECONDS = 5.


# objective 是已验证方案费用；reference 是独立参考记录，不用于生成候选。
# lagrangian 是 SA 之前拉格朗日阶段的记录：其 bound 是有效下界，只报告，不参与停止判断。
@dataclass
class Result:
    status: str = "RUNNING"
    message: str = ""
    objective: float = np.inf
    reference: dict = field(default_factory=dict)
    lagrangian: dict = field(default_factory=lambda: lagrangian.skipped("NOT_REACHED"))
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    w: np.ndarray | None = None
    scenarios: list = field(default_factory=list)
    validation: dict | None = None
    history: list = field(default_factory=list)
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


# 主流程：独立参考 → 初始化 → 拉格朗日阶段 → 分块搜索/建议 → 收尾建议 → 整理结果。
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
            run.whole.diagnostics['search_stop_reason'] = 'SA_BUDGET'
            run.whole.emit('sa_budget_exhausted')
        if result.status != "REFERENCE_TARGET_REACHED":
            _finish_advice(run)
    except Exception as error:
        # 只有截止时间信号归为 TIME_LIMIT；截止前后抛出的其他异常都是真实失败，保留阶段名。
        result.status = "TIME_LIMIT" if isinstance(error, TimeoutError) else f"{run.stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    finally:
        _shutdown(run)
    _finalize(run, d, scenarios)
    return result


# 连续松弛只用于初始化和初始乘子；初始方案不使用参考解。
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


# 块搜索：按小时分块，先由拉格朗日阶段得到乘子中心；块费用只对被松弛的连续耦合跨块行按该中心定价。
# 每块建立一个持久 BlockSearch，按轮依次访问全部块，至多 max_rounds 轮；价格固定，块内割和缓存一直有效。
def _search(run, m, relaxation):
    o, result, whole = run.o, run.result, run.whole
    if not o.sa_enabled:
        result.lagrangian = lagrangian.skipped("SA_DISABLED")
        whole.diagnostics['search_stop_reason'] = 'SA_DISABLED'
        return
    if result.status == "REFERENCE_TARGET_REACHED":
        result.lagrangian = lagrangian.skipped("REFERENCE_TARGET_REACHED")
        return
    with whole.sa_work():
        blocks, cross = partition(m, o.block_hours)
        center = _lagrangian(run, m, relaxation, blocks, cross)
        if result.status == "REFERENCE_TARGET_REACHED":
            return
        run.stage = "SEARCH_SETUP"
        prices = search_multipliers(m, cross, center)
        run.searches = [BlockSearch(priced(b, prices), o, j + 1, whole) for j, b in enumerate(blocks)]
    for k in range(1, o.max_rounds + 1):
        if _run_round(run, k):
            return
    whole.diagnostics['search_stop_reason'] = 'MAX_ROUNDS'


# 拉格朗日阶段：用精确块预言机和信赖域 bundle 最大化跨块对偶函数，返回乘子中心（初值为 LP 对偶）。
# 两次预言机调用之间轮询 LLM；可预期的降级只体现在 result.lagrangian 的状态与计数中，违反不变量抛错（LAGRANGIAN_FAILED）。
def _lagrangian(run, m, relaxation, blocks, cross):
    o, result, whole = run.o, run.result, run.whole
    initial = relaxation.prices[cross]
    if len(blocks) < 2:
        result.lagrangian = lagrangian.skipped("SINGLE_BLOCK")
        return initial
    run.stage = "LAGRANGIAN"
    state = lagrangian.start(len(blocks), m.eq[cross], initial)
    fixed = [whole.fixed[b.cols[b.binary]] for b in blocks]
    ended = perf_counter() + o.lagrangian_time_limit
    if o.verbose:
        print(f"[拉格朗日] 跨块约束 {len(cross)} 行（其中块搜索定价的连续耦合行 {int(relaxed_rows(m, cross).sum())} 行）；"
              f"预算 {o.lagrangian_time_limit:g} 秒，至多 {o.lagrangian_max_iterations} 次迭代；LP 松弛界 {relaxation.lower:.6f}。",
              flush=True)

    def remaining():
        return min(ended - perf_counter(), whole.remaining())

    def evaluate(j, lam, seconds):
        return lagrangian.oracle(blocks[j], fixed[j], lam, seconds, whole.best.point)

    # 在预言机调用之间推进 LLM；轮询期间的失败归到 ADVISOR 阶段。
    def poll():
        run.stage = "ADVISOR"
        whole.advice.advance([])
        run.stage = "LAGRANGIAN"
        return run.target_reached()

    def report(state):
        result.lagrangian = _lagrangian_record(state, relaxation, m, cross)
        if o.verbose and state.history:
            row = state.history[-1]
            print(f"[拉格朗日] 第 {row['iteration']} 次：{'严格步' if row['step'] == 'serious' else '空步'}，"
                  f"候选对偶值={row['value']:.6f}，中心对偶值={row['center_value']:.6f}，下界={row['bound']:.6f}，"
                  f"预测增益={row['predicted']:.6f}，盒半径={row['radius']:.6g}", flush=True)
        whole.emit("lagrangian_iteration")

    try:
        lagrangian.maximize(state, evaluate, iterations=o.lagrangian_max_iterations,
                            remaining=remaining, poll=poll, report=report)
    finally:
        if state.status == "RUNNING":
            state.status, state.message = "FAILED", "exception during the Lagrangian phase"
        result.lagrangian = _lagrangian_record(state, relaxation, m, cross)
        whole.timings["lagrangian"] += state.oracle_seconds + state.master_seconds
    whole.emit("lagrangian")
    if o.verbose:
        record = result.lagrangian
        print(f"[拉格朗日完成] 状态={record['status']} 下界={record['bound']} LP 松弛界={record['relaxation_bound']} "
              f"迭代={record['iterations']}（严格步 {record['serious_steps']}，空步 {record['null_steps']}） "
              f"乘子最大变化={record['multiplier_change']:.6g}，块搜索定价行={record['search_priced_rows']}", flush=True)
        if _lagrangian_degraded(record):
            print(f"[拉格朗日降级] 状态={record['status']} {record['message']} 预言机失败={record['oracle_failures']} "
                  f"LP 失败={record['master_failures']} 非精确预言机结果={record['inexact_oracle_results']}", flush=True)
    return state.center


# 拉格朗日记录加上 LP 松弛界、被松弛的连续耦合行数和块搜索实际定价的行数。
def _lagrangian_record(state, relaxation, m, cross):
    record = lagrangian.record(state)
    record.update(relaxation_bound=float(relaxation.lower) if np.isfinite(relaxation.lower) else None,
                  relaxed_rows=int(relaxed_rows(m, cross).sum()),
                  search_priced_rows=int(np.count_nonzero(search_multipliers(m, cross, state.center))))
    return record


# 收敛和迭代上限是正常结束；其他状态或任何失败、非精确结果都要醒目报告。
def _lagrangian_degraded(record):
    return (record["status"] not in ("CONVERGED", "ITERATION_LIMIT", "NOT_RUN")
            or bool(record.get("oracle_failures")) or bool(record.get("master_failures"))
            or bool(record.get("inexact_oracle_results")))


# 一个完整遍历轮：每块各采样一批。返回 True 表示达到参考目标、不再继续轮询。
def _run_round(run, k):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    start = _round_start(run)
    record = dict(round=k, complete=False, visited_blocks=[], started_seconds=perf_counter()-run.started)
    whole.diagnostics["rounds"].append(record)
    for j, search in enumerate(searches):
        whole.advice.advance(searches)
        if run.target_reached():
            break
        run.require_time()
        _visit_block(run, j, search, k, record)
        whole.advice.advance(searches)
    if result.status == 'REFERENCE_TARGET_REACHED':
        record.update(ended_seconds=perf_counter()-run.started, stop_reason='REFERENCE_TARGET_REACHED')
        whole.emit('reference_target_reached')
        return True
    return _finish_round(run, k, start, record)


# 在一个 SA 工作配额作用域内完成单块访问：块搜索，再把块内定价费用最低的块点拼回全天做 pool 候选。
def _visit_block(run, j, search, k, record):
    o, result, whole = run.o, run.result, run.whole
    with whole.sa_work():
        b = search.b
        run.stage = f"ROUND_{k}_BLOCK_{j + 1}"
        seeds = [whole.best.point[b.cols[b.binary]], *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
        search.step(seeds, k)
        run.require_time()
        record["visited_blocks"].append(j + 1)
        if search.best_local is not None:
            raw = result.u.copy(); raw[:, b.times] = search.best_local.point[b.u]
            proposal = {"proposal_id": f"pool:{k}:{j+1}", "round": k, "block": j + 1}
            x = whole.prepare(raw, "pool", proposal)
            whole.accept(x, "pool", proposal)
        if o.verbose:
            print(f"[分块进度] 轮次={k} 时间块={j + 1}/{len(run.searches)} "
                  f"全天可行成本(UB)={result.objective:.6f} "
                  f"参考成本={result.reference_cost}", flush=True)
        run.require_time()


# 完整遍历之后：更新 SA 轮计数、处理上界建议，最后刷新轮记录。
# 迭代记录须在 after_round 之前写入（它会进入 LLM 提示）；建议阶段可能新增割和评估，因此轮末再统计一次。
def _finish_round(run, k, start, record):
    o, result, whole, searches = run.o, run.result, run.whole, run.searches
    record.update(complete=True, sa_completed_seconds=perf_counter()-run.started,
                  incumbent_before=start.upper, incumbent_after_sa=result.objective)
    whole.diagnostics["complete_rounds"] += 1
    iteration = dict(iteration=len(whole.diagnostics["iterations"])+1, round=k,
                     **_round_progress(run, start), llm_failed=False)
    whole.diagnostics["iterations"].append(iteration)
    run.stage = "ADVISOR"
    whole.advice.after_round(start.upper, start.sa_calls, searches, k, k < o.max_rounds)
    iteration.update(**_round_progress(run, start),
                     llm_failed=any(row["status"] not in ("OK", "RUNNING", "CANCELLED")
                                    for row in result.llm_history[start.llm_calls:]))
    record.update(ended_seconds=perf_counter()-run.started, incumbent=result.objective,
                  llm_calls=result.llm_calls-start.llm_calls)
    whole.emit("round")
    for search in searches:
        search.summarize()
    run.require_time()
    return run.target_reached()


# 一轮开始时的快照：UB、SA 调用数、评估数、LLM 调用数和两类割的累计数量。
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
    return (sum(search.cut_totals["optimality"] for search in searches),
            sum(search.cut_totals["feasibility"] for search in searches))


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


# 独立复核最终方案，参考差距由 Result 随当前方案计算；有效拉格朗日下界不能高于已校验 UB，否则说明下界或 UB 有误。
def _finalize(run, d, scenarios):
    o, result, whole = run.o, run.result, run.whole
    for search in run.searches:
        search.summarize()
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if not result.validation['feasible']:
            result.status, result.message = "VALIDATION_FAILED", "Final schedule failed validation"
    bound = result.lagrangian.get("bound")
    if bound is not None and np.isfinite(result.objective):
        result.lagrangian.update(upper_at_end=result.objective,
                                 gap_percent_at_end=100*(result.objective-bound)/max(1., abs(result.objective)))
        if bound > result.objective + 1e-6*max(1., abs(result.objective)):
            result.status, result.message = "LAGRANGIAN_FAILED", "Lagrangian bound exceeds the validated UB"
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
        failures = result.diagnostics.get("dispatch_failures", 0)
        if failures or _lagrangian_degraded(result.lagrangian):
            print(f"[降级汇总] 调度数值失败 {failures} 次；拉格朗日状态 {result.lagrangian['status']}，"
                  f"预言机失败 {result.lagrangian.get('oracle_failures', {})}，LP 失败 {result.lagrangian.get('master_failures', {})}，"
                  f"非精确预言机结果 {result.lagrangian.get('inexact_oracle_results', 0)} 次。", flush=True)
