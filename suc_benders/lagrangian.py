"""Exact block Lagrangian oracle and a trust-region bundle for cross-block multipliers."""
# 模块职责：对给定跨块乘子精确求解各块拉格朗日子问题（Gurobi MIQP），并用盒式信赖域 Kelley 方法最大化对偶函数。
# 只有预言机返回的块最优点生成平面；可预期的降级以状态和计数返回，违反不变量直接抛错。
from collections import Counter
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .model import priced
from .optimization import optimize

# 实际增益不低于模型预测增益的这一比例时移动中心（serious step）。
SERIOUS_FRACTION = .1
# 模型预测增益低于 TOLERANCE*max(1,|中心值|) 且盒约束不起作用时判为收敛。
TOLERANCE = 1e-5
# 盒约束行对偶超过此值说明扩大盒子还能提高模型值，此时不能判为收敛。
BOX_DUAL_TOLERANCE = 1e-6
# 每块界从 Gurobi ObjBound 中扣除的安全余量。
BOUND_MARGIN = 1e-5
# 取整后的块点对块内约束允许的最大残差。
POINT_TOLERANCE = 1e-5
# 连续扩盒而不调用预言机的最大次数。
MAX_EXPANSIONS = 40
# 绝对值不超过此值的乘子视为数值噪声，不计入非零乘子数。
NONZERO = 1e-9


def _tolerance(value, relative=1e-6):
    return relative * max(1., abs(value))


# 块点 x 的拉格朗日平面：截距为原目标费用，斜率为该块在跨块行上的贡献；在任意 λ 处的值不低于 D_j(λ)。
def plane(b, x):
    return float(b.original_obj @ x + b.q @ x**2), np.asarray(b.crossA @ x - b.crossrhs)


# 精确求解块 b 在乘子 lam 下的拉格朗日子问题；fixed 为全天推出的块内固定位（NaN 为未定），作为变量上下界。
# incumbent 是已校验全天点，其块切片可行，用于检查界的上限。返回状态、是否精确、点值、界、平面和取整后的块点。
def oracle(b, fixed, lam, seconds, incumbent):
    p = priced(b, lam)
    lb, ub = b.lb.copy(), b.ub.copy()
    known = np.isfinite(fixed)
    lb[b.binary[known]] = ub[b.binary[known]] = fixed[known]
    raw = optimize(NS(**{**vars(p), "lb": lb, "ub": ub, "purpose": "lagrangian_block_miqp"}),
                   seconds, integral=True, int_feas_tol=1e-9)
    if raw.status in ("INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"):
        # 已校验 incumbent 的块切片满足全部块约束和固定位，子问题不可能不可行或无界。
        raise RuntimeError(f"Block Lagrangian subproblem {raw.status}")
    a, g = plane(b, incumbent[b.cols])
    reference = a + g @ p.lam
    bound = (raw.lower + p.constant - BOUND_MARGIN
             if np.isfinite(raw.lower) and abs(raw.lower) < 1e100 else -np.inf)
    if bound > reference + _tolerance(reference):
        raise RuntimeError("Block Lagrangian bound exceeds the incumbent slice")
    result = NS(status=raw.status, exact=False, value=np.inf, bound=bound, plane=None, point=None)
    if raw.x is None:
        return result
    x = raw.x.copy()
    binary = np.rint(x[b.binary])
    if np.abs(x[b.binary] - binary).max(initial=0) > 1e-6 or not np.array_equal(binary[known], fixed[known]):
        raise RuntimeError("Block Lagrangian point is not binary on its fixed bits")
    x[b.binary] = binary
    x = np.clip(x, lb, ub)
    residual = b.A @ x - b.rhs
    residual[b.eq] = np.abs(residual[b.eq])
    if residual.max(initial=0) > POINT_TOLERANCE:
        raise RuntimeError("Rounded block Lagrangian point violates block rows")
    result.point, result.plane = x, plane(b, x)
    result.value = result.plane[0] + result.plane[1] @ p.lam
    # 平面必须在生成它的乘子处与子问题目标一致，且不低于界。
    if abs(result.value - (raw.objval + p.constant)) > _tolerance(result.value):
        raise RuntimeError("Block Lagrangian plane is not tight at its multipliers")
    if bound > result.value + _tolerance(result.value):
        raise RuntimeError("Block Lagrangian bound exceeds its own point")
    result.exact = raw.status == "OPTIMAL"
    if result.exact and result.value > reference + _tolerance(reference, 2e-6):
        raise RuntimeError("Optimal block Lagrangian value exceeds the incumbent slice")
    return result


# 建立 bundle 状态：中心从 initial（长度同 cross）开始，平面按块以取整后的 (截距, 斜率) 去重。
def start(blocks, equality, initial):
    initial = np.asarray(initial, float).copy()
    return NS(equality=np.asarray(equality, bool), initial=initial, center=initial.copy(),
              center_value=-np.inf, initial_value=None, initial_bound=None, bound=-np.inf,
              radius=max(1., float(np.abs(initial).max(initial=0.))), planes=[{} for _ in range(blocks)],
              status="RUNNING", message="", iterations=0, serious=0, null=0, expansions=0, oracle_calls=0,
              inexact=0, oracle_failures=Counter(), master_failures=Counter(), duplicates=0, history=[],
              oracle_seconds=0., master_seconds=0., pass_seconds=0.)


# 当前切平面模型在 lam 处的值：各块取其平面的最小值再求和。
def model_value(state, lam):
    return sum(min(a + g @ lam for a, g in group.values()) for group in state.planes)


# 在以中心为心、半径为 radius 的盒内最大化切平面模型；binding 表示盒约束行对偶非零，扩大盒子仍可提高模型值。
# 不等式乘子的非负约束是变量下界，不算盒约束。
def trust_region(state, seconds):
    n, B = len(state.center), len(state.planes)
    rows, rhs = [], []
    for j, group in enumerate(state.planes):
        for a, g in group.values():
            theta = np.zeros(B); theta[j] = 1.
            rows.append(np.r_[-g, theta]); rhs.append(a)
    c, radius = state.center, state.radius
    lower = state.equality | (c - radius > 0)
    box = sparse.vstack([sparse.hstack([sparse.eye(n), sparse.csr_matrix((n, B))]),
                         sparse.hstack([-sparse.eye(n, format="csr")[lower], sparse.csr_matrix((lower.sum(), B))])])
    m = NS(A=sparse.vstack([sparse.csr_matrix(np.asarray(rows)), box], format="csr"),
           rhs=np.r_[rhs, c + radius, -(c - radius)[lower]], eq=np.zeros(len(rhs) + n + lower.sum(), bool),
           obj=np.r_[np.zeros(n), -np.ones(B)], q=np.zeros(n + B),
           lb=np.r_[np.where(state.equality, -np.inf, 0.), np.full(B, -np.inf)], ub=np.full(n + B, np.inf),
           binary=[], purpose="lagrangian_trust_region_lp")
    raw = optimize(m, seconds)
    if raw.status != "OPTIMAL":
        return NS(status=raw.status, candidate=None, model=np.nan, binding=None)
    candidate = raw.x[:n].copy()
    candidate[~state.equality] = np.maximum(0., candidate[~state.equality])
    return NS(status="OPTIMAL", candidate=candidate, model=-raw.objval,
              binding=bool(raw.prices[len(rhs):].max(initial=0.) > BOX_DUAL_TOLERANCE))


# 在 lam 处逐块调用预言机并收录平面。返回 (各块结果, None) 或 (None, 停止状态)；时间不足、中断或预言机失败都不产生部分结果。
def _pass(state, lam, evaluate, remaining, poll):
    results, started = [], perf_counter()
    for j in range(len(state.planes)):
        if poll():
            return None, "INTERRUPTED"
        seconds = remaining()
        if seconds <= 0:
            return None, "TIME_LIMIT"
        r = evaluate(j, lam, seconds)
        state.oracle_calls += 1
        if r.plane is None:
            if r.status == "TIME_LIMIT":
                return None, "TIME_LIMIT"
            state.oracle_failures[r.status] += 1
            return None, f"ORACLE_{r.status}"
        a, g = r.plane
        key = np.round(np.r_[a, g], 6).tobytes()
        if key in state.planes[j]:
            state.duplicates += 1
        state.planes[j][key] = (a, g)
        state.inexact += not r.exact
        results.append(r)
    state.oracle_seconds += perf_counter() - started
    state.iterations += 1
    bounds = [r.bound for r in results]
    if np.isfinite(bounds).all():
        state.bound = max(state.bound, float(sum(bounds)))
    # 每个已存平面都是 D_j 的上估计，不能低于本次得到的块界。
    for j, r in enumerate(results):
        lowest = min(a + g @ lam for a, g in state.planes[j].values())
        if np.isfinite(r.bound) and lowest < r.bound - _tolerance(r.bound):
            raise RuntimeError("A stored Lagrangian plane lies below the block bound")
    return results, None


# 从中心出发最大化对偶函数，直到收敛、达到迭代或时间上限、被 poll 中断或出现降级。
# evaluate(j, lam, seconds) 返回 oracle() 形式的结果；remaining() 给出剩余秒数；report(state) 在每次迭代后调用。
def maximize(state, evaluate, *, iterations, remaining, poll=lambda: False, report=lambda state: None):
    if iterations <= 0:
        state.status = "NOT_RUN"; state.message = "ZERO_ITERATIONS"
        return state
    before = remaining()
    results, stop = _pass(state, state.center, evaluate, remaining, poll)
    if stop is None and not all(r.exact for r in results):
        stop = "TIME_LIMIT"
    if stop is not None:
        state.status = stop
        return state
    state.pass_seconds = before - remaining()
    state.center_value = state.initial_value = float(sum(r.value for r in results))
    state.initial_bound = state.bound
    report(state)
    expansions = 0
    while True:
        if state.iterations >= iterations:
            state.status = "ITERATION_LIMIT"; break
        if remaining() <= state.pass_seconds:
            state.status = "TIME_LIMIT"; break
        if poll():
            state.status = "INTERRUPTED"; break
        started = perf_counter()
        tr = trust_region(state, remaining())
        state.master_seconds += perf_counter() - started
        if tr.status != "OPTIMAL":
            state.master_failures[tr.status] += 1
            state.status = "TIME_LIMIT" if tr.status == "TIME_LIMIT" else f"MASTER_{tr.status}"; break
        predicted = tr.model - state.center_value
        if predicted <= TOLERANCE * max(1., abs(state.center_value)):
            if not tr.binding:
                state.status = "CONVERGED"; break
            # 盒约束起作用时，小的预测增益只说明盒子太小，扩盒后重解模型，不调用预言机。
            expansions += 1; state.expansions += 1; state.radius *= 2
            if expansions > MAX_EXPANSIONS:
                state.status = "TRUST_REGION_LIMIT"; break
            continue
        expansions = 0
        before = remaining()
        results, stop = _pass(state, tr.candidate, evaluate, remaining, poll)
        if stop is not None:
            state.status = stop; break
        state.pass_seconds = before - remaining()
        value = float(sum(r.value for r in results))
        exact = all(r.exact for r in results)
        serious = exact and value - state.center_value >= SERIOUS_FRACTION * predicted
        if serious:
            state.center, state.center_value = tr.candidate, value
            state.serious += 1; state.radius *= 2
        else:
            state.null += 1; state.radius /= 2
        state.history.append(dict(iteration=state.iterations, step="serious" if serious else "null",
            value=value, center_value=state.center_value, model=tr.model, predicted=predicted,
            radius=state.radius, bound=state.bound, exact=exact))
        report(state)
        if not exact:
            state.status = "TIME_LIMIT"; break
    return state


# 可落盘的 bundle 记录；history 可按需去掉以保持摘要精简。
def record(state, history=True):
    row = dict(status=state.status, message=state.message, iterations=state.iterations,
               serious_steps=state.serious, null_steps=state.null, expansions=state.expansions,
               oracle_calls=state.oracle_calls, inexact_oracle_results=state.inexact,
               oracle_failures=dict(state.oracle_failures), master_failures=dict(state.master_failures),
               planes=sum(len(group) for group in state.planes), duplicate_planes=state.duplicates,
               multipliers=len(state.center), nonzero_multipliers=int(np.count_nonzero(np.abs(state.center) > NONZERO)),
               multiplier_change=float(np.abs(state.center - state.initial).max(initial=0.)), radius=state.radius,
               initial_value=state.initial_value, initial_bound=state.initial_bound,
               center_value=state.center_value if np.isfinite(state.center_value) else None,
               bound=state.bound if np.isfinite(state.bound) else None,
               oracle_seconds=state.oracle_seconds, master_seconds=state.master_seconds)
    if history:
        row["history"] = list(state.history)
    return row


# 未运行时的记录，reason 说明跳过原因。
def skipped(reason):
    return dict(status="NOT_RUN", message=reason, iterations=0, serious_steps=0, bound=None)
