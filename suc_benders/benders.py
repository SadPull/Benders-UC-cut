"""Continuous recourse cuts, LP master bounds and dual coordination."""
# 模块职责：共享离散可行性检查、连续调度割、块主问题松弛及拉格朗日协调。
# X/binary 为 u/v/w 等二进制列，Y/continuous 为连续列；不等式统一采用 A @ x <= rhs。
from fractions import Fraction
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


# 把离散约束、容量/AND 关系与已有 Benders 割组装成线性主问题，并可选求其连续松弛下界。
# 末列 theta 表示不含 b.constant 的块总费用下估计；返回的松弛解不用于替代 SA 候选。
def master(b, cuts, seconds=0, *, search_reachability=False):
    X, Y = b.binary, b.continuous
    nx = len(X)
    hard = b.A[:, Y].getnnz(axis=1) == 0
    A = b.A[hard][:, X].tocsr()
    rows = [dict(zip(A.indices[A.indptr[j]:A.indptr[j + 1]], A.data[A.indptr[j]:A.indptr[j + 1]])) for j in range(A.shape[0])]
    rhs, eq = list(b.rhs[hard]), list(b.eq[hard])
    kind = ["hard"] * len(rows)
    mapping = np.full(len(b.obj), -1); mapping[X] = np.arange(nx)
    pairs, nb, capacity_rows = [], nx, {}

    # 添加主问题约束及 hard/capacity/and/optimality/feasibility 标签，供 QUBO 选择对应编码。
    def add(indices, values, bound, equality=False, label="hard"):
        rows.append({int(j): float(v) for j, v in zip(indices, values) if v != 0})
        rhs.append(bound); eq.append(equality); kind.append(label)

    if search_reachability or hasattr(b, 'search_fixed'):
        start, stop = transition_allowed(b.d)
        forbidden = np.r_[b.v[~start].ravel(), b.w[~stop[:, b.times]]]
        for j in forbidden:
            add([mapping[j]], [1.], 0., True)
    # Whole-horizon implications survive decomposition at a block boundary.
    for j in np.flatnonzero(np.isfinite(getattr(b, 'search_fixed', []))):
        add([j], [1.], float(b.search_fixed[j]), True)

    # 用精确容量系数保留启停限制；min_up=1 时 AND 辅助位修正同小时启停容量的重叠。
    for t, hour in enumerate(b.times):
        capacity = {}
        for i in range(b.d.n_units):
            u, v = mapping[b.u[i, t]], mapping[b.v[i, t]]
            pm = Fraction(str(b.d.pmax[i]))
            sr, dr = pm - Fraction(str(b.d.pstart[i])), pm - Fraction(str(b.d.pshut[i]))
            capacity[u] = pm; capacity[v] = -sr
            if t < len(b.times) - 1:
                w = mapping[b.w[i, t + 1]]; capacity[w] = -dr
                if b.d.min_up[i] == 1:
                    z = nb; nb += 1; capacity[z] = min(sr, dr)
                    pairs.append((z, v, w))
                    add([z, v], [1, -1], 0, label="and"); add([z, w], [1, -1], 0, label="and")
                    add([v, w, z], [1, 1, -1], 1, label="and")
        required = (1 + Fraction(str(b.d.reserve_fraction))) * max(Fraction(str(s.load[hour])) for s in b.scenarios)
        capacity_rows[len(rows)] = ({j: -v for j, v in capacity.items()}, required)
        add(capacity.keys(), [-v for v in capacity.values()], -float(required), label="capacity")
    # theta 下估计二进制固定费用加连续补偿；b.constant 保留在块模型之外，汇总时加回。
    theta = nb
    qlo = np.minimum(b.obj[Y] * b.lb[Y], b.obj[Y] * b.ub[Y]).sum()
    add([*range(nx), theta], [*b.obj[X], -1], -qlo, label="optimality")
    for c in cuts:
        if c.optimality:
            add([*range(nx), theta], [*(c.beta + b.obj[X]), -1], -c.alpha, label="optimality")
        else:
            add(range(nx), c.beta, -c.alpha, label="feasibility")
    lo = np.minimum(b.obj * b.lb, b.obj * b.ub).sum()
    hi = np.sum(np.maximum(b.obj * b.lb, b.obj * b.ub) + b.q * b.ub ** 2)
    I, J, V = [], [], []
    for i, row in enumerate(rows):
        I.extend([i] * len(row)); J.extend(row.keys()); V.extend(row.values())
    m = NS(A=sparse.coo_matrix((V, (I, J)), shape=(len(rows), nb + 1)).tocsr(),
           rhs=np.asarray(rhs), eq=np.asarray(eq, bool), obj=np.r_[np.zeros(nb), 1.],
           q=np.zeros(nb + 1), lb=np.r_[np.zeros(nb), lo], ub=np.r_[np.ones(nb), hi],
           binary=np.arange(nb), nx=nx, and_pairs=pairs, kind=kind, scale=b.scale,
           capacity_rows=capacity_rows, constant=0.)
    m.purpose = 'master_lp_relaxation'
    lower = -np.inf
    if seconds > 0:
        raw = optimize(m, seconds)
        if raw.status == "INFEASIBLE":
            raise RuntimeError("LP master is infeasible")
        if raw.status == "OPTIMAL":
            lower = (raw.objval) + b.constant - 1e-5
    # Relaxation solutions are bounds only, not SA candidates.
    return m, lower


# 用各块的可行费用平面优化跨块乘子，返回新乘子、协调对偶模型的上界与处理状态。
# 主问题被 incumbent 截顶或失败时保留 previous；L1 选择失败则回退主问题乘子，均不中止搜索。
def coordinate(planes, equality, incumbent, seconds, previous):
    started = perf_counter()
    nc, B = len(equality), len(planes)
    rows, rhs = [], []
    for j, group in enumerate(planes):
        if not group:
            raise ValueError("Each block needs a feasible plane")
        for intercept, slope in group:
            row = sparse.lil_matrix((1, nc + B))
            row[0, :nc] = -slope; row[0, nc + j] = 1
            rows.append(row.tocsr()); rhs.append(intercept)
    rows.append(sparse.csr_matrix(np.r_[np.zeros(nc), np.ones(B)][None, :]))
    rhs.append(incumbent)
    m = NS(A=sparse.vstack(rows, format="csr"), rhs=np.asarray(rhs), eq=np.zeros(len(rhs), bool),
           obj=np.r_[np.zeros(nc), -np.ones(B)], q=np.zeros(nc + B),
           lb=np.r_[np.where(equality, -np.inf, 0), np.full(B, -np.inf)],
           ub=np.full(nc + B, np.inf), binary=[])
    m.purpose = 'coordination_lp'
    previous = np.asarray(previous, float).copy()
    raw = optimize(m, seconds)
    if raw.status != "OPTIMAL":
        return previous, np.inf, dict(status=f"PRIMARY_{raw.status}")
    upper = -raw.objval
    # 截顶时最优面由 incumbent 决定，乘子不含平面信息；目标带同一容差，避免与截顶行矛盾。
    tolerance = max(1e-6 * abs(incumbent), 1e-4)
    if upper >= incumbent - tolerance:
        return previous, upper, dict(status="CAPPED")
    info = dict(status="OPTIMAL")
    if nc:
        # The primary optimum can have unbounded flat multiplier directions.
        # Select the smallest L1 multipliers within a tolerance of its upper bound.
        primary = raw
        identity = sparse.eye(nc, format="csr")
        absolute = sparse.hstack([identity, sparse.csr_matrix((nc, B))], format="csr")
        target = sparse.csr_matrix(np.r_[np.zeros(nc), -np.ones(B), np.zeros(nc)][None, :])
        stable = NS(A=sparse.vstack([
            sparse.hstack([m.A, sparse.csr_matrix((len(m.rhs), nc))]), target,
            sparse.hstack([absolute, -identity]), sparse.hstack([-absolute, -identity])], format="csr"),
            rhs=np.r_[m.rhs, -(upper - tolerance), np.zeros(2*nc)],
            eq=np.zeros(len(m.rhs)+1+2*nc, bool), obj=np.r_[np.zeros(nc+B), np.ones(nc)],
            q=np.zeros(2*nc+B), lb=np.r_[m.lb, np.zeros(nc)], ub=np.r_[m.ub, np.full(nc, np.inf)], binary=[])
        stable.purpose = 'coordination_minimum_l1_lp'
        remaining = seconds-(perf_counter()-started)
        raw = optimize(stable, remaining) if remaining > 0 else NS(status="TIME_LIMIT")
        if raw.status != "OPTIMAL":
            info = dict(status="SELECTION_FALLBACK", selection_status=raw.status)
            raw = primary
    prices = raw.x[:nc].copy()
    prices[~equality] = np.maximum(0, prices[~equality])
    return prices, upper, info
