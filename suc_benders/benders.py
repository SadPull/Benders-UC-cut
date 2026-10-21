"""Discrete feasibility, continuous recourse cuts and block hard rows."""
# 模块职责：共享离散可行性检查、固定二进制后的连续调度与 Benders 割，以及块内硬约束行。
# X/binary 为 u/v/w 等二进制列，Y/continuous 为连续列；不等式统一采用 A @ x <= rhs。
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .optimization import optimize
from .validation import transition_allowed


# 检查只含二进制变量的约束和严格 0/1 输入；连续调度行留给子问题判断。
# hard 可复用调用方已计算的行掩码，等式残差取绝对值，容差为 1e-7。
def discrete_feasible(b, x, hard=None):
    if hard is None:
        hard = b.A[:, b.continuous].getnnz(axis=1) == 0
    residual = b.A[hard][:, b.binary] @ x - b.rhs[hard]
    residual[b.eq[hard]] = np.abs(residual[b.eq[hard]])
    return bool(np.isin(x, [0, 1]).all() and residual.max(initial=0) <= 1e-7)


# 固定精确二进制向量 x 后求解连续调度，返回可行性、实际费用、点、对偶价格与割。
# certified=True 时使用带安全余量的对偶下界或可验证分离式生成割；不可行探测也可产出割。
def dispatch(b, x, seconds, *, certified=False):
    started = perf_counter()
    X, Y = b.binary, b.continuous
    x = np.asarray(x)
    if x.shape != (len(X),) or not np.isin(x, [0, 1]).all():
        raise ValueError("Dispatch requires an exact binary vector")
    result = NS(feasible=False, status="DISCRETE_INFEASIBLE", cost=np.inf,
                point=None, cut=None, prices=None)
    hard = b.A[:, Y].getnnz(axis=1) == 0
    if not discrete_feasible(b, x, hard):
        return result
    rows = np.flatnonzero(~hard)
    B = b.A[rows][:, X]
    g = NS(A=b.A[rows][:, Y], rhs=b.rhs[rows] - B @ x, eq=b.eq[rows],
           obj=b.obj[Y], q=b.q[Y], lb=b.lb[Y], ub=b.ub[Y], binary=[], purpose='fixed_commitment_qp')
    raw = optimize(g, seconds)
    result.status = raw.status
    if raw.status == "OPTIMAL":
        # 连续补偿函数的下估计为 alpha + beta @ x；固定二进制费用在总成本中另外加入。
        beta = np.asarray(B.T @ raw.prices)
        anchor = raw.objval
        if certified:
            if not np.isfinite(raw.lower) or raw.lower > raw.objval + 1e-5:
                raise RuntimeError("Invalid recourse dual bound")
            anchor = raw.lower - 1e-5
        result.cut = NS(beta=beta, alpha=anchor - beta @ x, optimality=True, certified=certified)
        result.point = np.zeros(len(b.obj)); result.point[X] = x; result.point[Y] = raw.x
        result.cost = raw.objval + b.obj[X] @ x + b.constant
        result.feasible = True
        result.prices = np.zeros(len(b.rhs)); result.prices[rows] = raw.prices
    elif raw.status in ("INFEASIBLE", "INF_OR_UNBD") and perf_counter() - started < seconds:
        # 用非负弹性松弛寻找分离价格；等式展开成两个方向，但仍共用原行的松弛变量。
        E = np.flatnonzero(g.eq)
        C = sparse.vstack([g.A, -g.A[E]], format="csr")
        D = sparse.vstack([B, -B[E]], format="csr")
        h = np.r_[b.rhs[rows], -b.rhs[rows[E]]]
        nr, ny, ns = len(h), len(Y), len(rows)
        slack = sparse.eye(ns, format="csr")
        slack = sparse.vstack([slack, slack[E]], format="csr")
        elastic = NS(A=sparse.hstack([C, -slack], format="csr"), rhs=h - D @ x,
                     eq=np.zeros(nr, bool), obj=np.r_[np.zeros(ny), np.ones(ns)],
                     q=np.zeros(ny + ns), lb=np.r_[g.lb, np.zeros(ns)],
                     ub=np.r_[g.ub, np.full(ns, np.inf)], binary=[])
        elastic.purpose = 'feasibility_lp'
        raw = optimize(elastic, seconds - (perf_counter() - started))
        result.status = raw.status
        if raw.status == "OPTIMAL":
            prices = np.clip(raw.prices, 0, 1)
            beta = np.asarray(D.T @ prices)
            alpha = float(raw.objval - beta @ x)
            if certified:
                # 裁剪价格后重新在原有界连续域上计算截距，认证不能直接沿用旧弹性目标值。
                # For every feasible y, prices @ (C*y + D*x - h) <= 0.
                # Minimize C.T @ prices over the original bounded y domain.
                # This remains valid after projecting numerical multipliers.
                c = np.asarray(C.T @ prices)
                alpha = float(np.minimum(c * g.lb, c * g.ub).sum() - prices @ h - 1e-5)
                if not np.isfinite(alpha) or alpha + beta @ x <= 1e-7:
                    raise RuntimeError("Feasibility cut does not certify separation")
            result.cut = NS(beta=beta, alpha=alpha, optimality=False, certified=certified)
            result.status = "INFEASIBLE"
    return result


# 汇集块内只含二进制列的硬约束行，供固定位传播与搜索 QUBO 使用；列为块内二进制位（顺序同 b.binary）。
# reachability=True 或给出 fixed 时加入不可达启停的禁止行；fixed 为块内二进制位的已知取值（NaN 表示未定），
# 以等式行固定，使全天推出的固定位在块边界处仍然成立。
def hard_rows(b, *, reachability=False, fixed=None):
    X, Y = b.binary, b.continuous
    nx = len(X)
    hard = b.A[:, Y].getnnz(axis=1) == 0
    mapping = np.full(len(b.obj), -1); mapping[X] = np.arange(nx)
    single = []
    if reachability or fixed is not None:
        start, stop = transition_allowed(b.d)
        single += [(mapping[j], 0.) for j in np.r_[b.v[~start].ravel(), b.w[~stop[:, b.times]]]]
    if fixed is not None:
        single += [(j, float(fixed[j])) for j in np.flatnonzero(np.isfinite(fixed))]
    columns = np.asarray([j for j, _ in single], int)
    extra = sparse.csr_matrix((np.ones(len(single)), (np.arange(len(single)), columns)), shape=(len(single), nx))
    return NS(A=sparse.vstack([b.A[hard][:, X], extra], format="csr"),
              rhs=np.r_[b.rhs[hard], [value for _, value in single]],
              eq=np.r_[b.eq[hard], np.ones(len(single), bool)], binary=np.arange(nx), nx=nx)
