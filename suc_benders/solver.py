"""Coordinate time-block search, upper advice and independent certification."""
# 主循环只安排各阶段的执行顺序、预算与结束状态；候选处理在 search 中。
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .benders import coordinate
from .model import model, partition
from .optimization import optimize
from .search import BlockSearch, WholeDay, SABudgetExhausted, closed
from .validation import validate


# 可序列化求解结果：objective 是已验证方案费用，lower_bound 是当前模式认可的全局下界。
# coordination_upper 属于协调对偶模型；状态、原问题 gap 与诊断记录需分别解释。
# 求解中 status 为 RUNNING；结束时只会是 CONVERGED、TIME_LIMIT 或 <阶段>_FAILED。
@dataclass
class Result:
    status: str = "RUNNING"
    message: str = ""
    objective: float = np.inf
    lower_bound: float = -np.inf
    relative_gap: float = np.inf
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
    certification_history: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.


# 主流程：初始化/恢复 → 固定价格轮询所有块 → 轮末建议/认证 → 协调更新价格 → 整理结果。
# 累计历史执行时间计入预算；保留阶段名定位失败，并在最终返回前独立复核全天方案。
def solve(d, scenarios, o, *, resume_state=None, progress=None, started_at=None):
    prior = resume_state["charged_elapsed_seconds"] if resume_state is not None else 0.
    if not np.isfinite(prior) or prior < 0 or prior >= o.time_limit:
        raise ValueError("Invalid or exhausted cumulative resume budget")
    started = (perf_counter() if started_at is None else started_at)-prior; deadline = started + o.time_limit
    deadline -= 5.
    result, whole = Result(), None
    searches = []
    stage = "INITIALIZATION"

    # 在 WholeDay 尚未构造时使用本地 deadline，构造后转用当前工作配额调整后的 deadline。
    def remaining():
        return whole.remaining() if whole is not None else deadline - perf_counter()

    try:
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        full.audit_cuts = bool(getattr(progress, 'audit_cuts', False))
        whole = WholeDay(m, full, o, deadline, result, started, progress)
        phase = perf_counter()
        relaxation = optimize(m, remaining())
        whole.timings["initialization"] += perf_counter() - phase
        if remaining() <= 0:
            raise TimeoutError
        if relaxation.status != "OPTIMAL":
            raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
        if hasattr(progress, 'relaxation'):
            progress.relaxation(m, relaxation, perf_counter()-started)
        result.lower_bound = relaxation.lower
        if not np.isfinite(result.lower_bound):
            raise RuntimeError("Continuous relaxation has no finite certified lower bound")
        prices = relaxation.prices
        if resume_state is not None:
            stage = "RESUME"
            whole.restore(resume_state)
        else:
            whole.initialize(relaxation)
        if whole.best is None:
            raise RuntimeError("No validated initial schedule")
        whole.emit("initialization")
        whole.certifier.start()
        if resume_state is None:
            whole.advice.kickoff()
        if not o.sa_enabled and whole.certifier.converged():
            result.status = "CONVERGED"
        try:
            with whole.sa_work():
                blocks, cross = partition(m, o.block_hours, prices) if o.sa_enabled else ([], np.array([], int))
            planes, previous = [[] for _ in blocks], [[] for _ in blocks]
            first_outer, first_round = whole.resume_cursor["outer"], whole.resume_cursor["round"]
            for outer in (range(first_outer, o.max_outer + 1) if o.sa_enabled and result.status != "CONVERGED" else []):
                if remaining() <= 0:
                    raise TimeoutError
                with whole.sa_work():
                    # 新乘子对应新的块费用与局部割环境，历史二进制种子必须重新调度。
                    blocks, _ = partition(m, o.block_hours, prices)
                    searches = [BlockSearch(b, o, outer, j + 1, whole, planes[j]) for j, b in enumerate(blocks)]
                if whole.advice.state["phase"] == "llm":
                    stage = "ADVISOR_RESUME"
                    whole.advice.advance(searches)
                    if whole.certifier.converged():
                        result.status = "CONVERGED"
                        break
                # 每个 inner 是完整遍历轮；尚未闭合的块各最多采样一批，已闭合块跳过采样。
                for inner in range(first_round if outer == first_outer else 1, o.max_inner + 1):
                    before = result.objective
                    sa_before = whole.diagnostics["sa_calls"]
                    before_count, before_calls = len(whole.diagnostics["evaluations"]), result.llm_calls
                    before_opt = sum(c.optimality for search in searches for c in search.cuts)
                    before_feas = sum(not c.optimality for search in searches for c in search.cuts)
                    global_round = whole.diagnostics['complete_rounds']+1
                    round_record = dict(outer=outer, round=inner, global_round=global_round,
                                        complete=False, visited_blocks=[],
                                        skipped_closed_blocks=[], started_seconds=perf_counter()-started)
                    whole.diagnostics["rounds"].append(round_record)
                    for j, search in enumerate(searches):
                        whole.advice.advance(searches)
                        whole.certifier.step()
                        if whole.certifier.converged():
                            result.status = 'CONVERGED'
                            break
                        if remaining() <= 0:
                            raise TimeoutError
                        if closed(search.lower, search.upper, o.tolerance):
                            round_record["skipped_closed_blocks"].append(j + 1)
                        else:
                            with whole.sa_work():
                                b = search.b
                                stage = f"BLOCK_{outer}_{j + 1}_ROUND_{inner}"
                                seeds = [whole.best.point[b.cols[b.binary]], *(previous[j] if inner == 1 else []),
                                         *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
                                search.step(seeds, inner)
                                if remaining() <= 0:
                                    raise TimeoutError
                                round_record["visited_blocks"].append(j + 1)
                                if search.best is not None:
                                    raw = result.u.copy(); raw[:, b.times] = search.best.point[b.u]
                                    record = {"proposal_id": f"pool:{outer}:{inner}:{j+1}",
                                              "outer": outer, "round": inner, "block": j + 1}
                                    x = whole.prepare(raw, "pool", record)
                                    whole.accept(x, "pool", record)
                                if o.verbose:
                                    print(f"[分块进度] 外层={outer} 内层轮次={inner} 时间块={j + 1}/{len(blocks)} "
                                          f"块内[下界, 上界]=[{search.lower:.6f}, {search.upper:.6f}] "
                                          f"全天可行成本(UB)={result.objective:.6f} "
                                          f"认证下界(LB)={result.lower_bound:.6f}", flush=True)
                                if remaining() <= 0:
                                    raise TimeoutError
                        whole.advice.advance(searches)
                    if result.status == 'CONVERGED':
                        round_record.update(ended_seconds=perf_counter()-started, stop_reason='CONVERGED')
                        whole.emit('convergence')
                        break
                    # 完整遍历才更新 SA 轮计数和轮末停滞窗口。
                    round_record.update(complete=True, sa_completed_seconds=perf_counter()-started,
                                        incumbent_before=before, incumbent_after_sa=result.objective)
                    whole.diagnostics["complete_rounds"] += 1
                    iteration = dict(iteration=len(whole.diagnostics["iterations"])+1, outer=outer, round=inner,
                        incumbent=result.objective,
                        optimality_cuts_added=sum(c.optimality for x in searches for c in x.cuts)-before_opt,
                        feasibility_cuts_added=sum(not c.optimality for x in searches for c in x.cuts)-before_feas,
                        new_candidates=len(whole.diagnostics["evaluations"])-before_count, llm_failed=False)
                    whole.diagnostics["iterations"].append(iteration)
                    all_closed = all(closed(x.lower, x.upper, o.tolerance) for x in searches)
                    whole.resume_cursor = (dict(outer=outer+1, round=1) if all_closed or inner == o.max_inner
                                           else dict(outer=outer, round=inner+1))
                    # 数值认证与上界建议都位于完整 SA 轮之后。
                    stage = "CERTIFICATION"
                    whole.certifier.step()
                    stage = "ADVISOR"
                    whole.advice.after_round(before, sa_before, searches, outer, inner,
                                             not (outer == o.max_outer and inner == o.max_inner))
                    iteration.update(incumbent=result.objective,
                        optimality_cuts_added=sum(c.optimality for x in searches for c in x.cuts)-before_opt,
                        feasibility_cuts_added=sum(not c.optimality for x in searches for c in x.cuts)-before_feas,
                        new_candidates=len(whole.diagnostics["evaluations"])-before_count,
                        llm_failed=any(row["status"] not in ("OK", "RUNNING", "CANCELLED")
                                       for row in result.llm_history[before_calls:]))
                    round_record.update(ended_seconds=perf_counter()-started, incumbent=result.objective,
                                        llm_calls=result.llm_calls-before_calls)
                    whole.emit("round")
                    for search in searches:
                        search.summarize()
                    if remaining() <= 0:
                        raise TimeoutError
                    if whole.certifier.converged():
                        result.status = "CONVERGED"
                        break
                    if all_closed:
                        break
                if result.status == "CONVERGED":
                    break
                whole.advice.advance(searches)
                with whole.sa_work():
                    previous = [search.seeds for search in searches]
                    stage = "COORDINATION"
                    phase = perf_counter()
                    values, upper, info = coordinate(planes, m.eq[cross], result.objective, remaining(),
                                                     prices[cross])
                    whole.timings["coordination"] += perf_counter() - phase
                    if remaining() <= 0:
                        raise TimeoutError
                    result.coordination_upper = min(result.coordination_upper, upper)
                    result.outer_history.append({"outer": outer, "lower": result.lower_bound, "coordination_upper": upper,
                                                 "coordination": info, "complete_rounds": inner,
                                                 "unconverged_blocks": [x.index for x in searches
                                                     if not closed(x.lower, x.upper, o.tolerance)]})
                    whole.diagnostics["coordination_count"] += 1
                    whole.emit("coordination")
                    dual_lower = max(relaxation.lower, float(sum(search.lower for search in searches)))
                    result.outer_history[-1]["dual_lower"] = dual_lower
                    result.outer_history[-1]["dual_converged"] = closed(dual_lower, upper, o.tolerance, dual_lower)
                    prices = np.zeros(len(m.rhs)); prices[cross] = values
        except SABudgetExhausted:
            whole.diagnostics['sa_budget_exhausted'] = True
            whole.emit('sa_budget_exhausted')
        if result.status != "CONVERGED":
            # SA 达到迭代或配额上限后，保留已有方案与割，继续使用剩余总预算进行数值认证。
            stage = "CERTIFICATION"
            while remaining() > 0 and not whole.certifier.converged():
                whole.certifier.step()
                whole.advice.advance(searches)
            if remaining() <= 0:
                raise TimeoutError
            result.status = "CONVERGED"
    except Exception as error:
        result.status = "TIME_LIMIT" if remaining() <= 0 or isinstance(error, TimeoutError) else f"{stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    finally:
        if whole is not None:
            whole.advice.close()
            try:
                whole.certifier.close()
            except Exception as error:
                whole.diagnostics['proof_shutdown_error'] = type(error).__name__
                result.status, result.message = 'CERTIFICATION_FAILED', str(error)
    for search in searches:
        search.summarize()
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if result.lower_bound > result.objective + 1e-5:
            result.status, result.message = "CERTIFICATION_FAILED", "Lower bound exceeds validated cost"
            result.relative_gap = (result.objective - result.lower_bound) / max(1., abs(result.objective))
        else:
            result.relative_gap = max(0., result.objective - result.lower_bound) / max(1., abs(result.objective))
    if whole is not None:
        whole.diagnostics['search_statistics'] = whole.search_statistics()
    result.elapsed_seconds = perf_counter() - started
    if whole is not None:
        whole.diagnostics['search_statistics']['new_feasible_per_second'] = (
            whole.diagnostics['search_statistics']['new_feasible']/max(result.elapsed_seconds, 1e-9))
        whole.timings["other"] = max(0., result.elapsed_seconds - sum(whole.timings.values()))
        result.diagnostics.update(evaluation_count=len(whole.cache), llm_call_count=result.llm_calls,
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit),
                                  resume_cursor=whole.resume_cursor)
        whole.emit("final")
    if o.verbose:
        print(f"[主算法结束] 状态={result.status} 可行成本(UB)={result.objective:.6f}, "
              f"认证下界(LB)={result.lower_bound:.6f}, "
              f"认证间隙={100 * result.relative_gap:.4f}%, 耗时={result.elapsed_seconds:.2f}秒", flush=True)
    return result
