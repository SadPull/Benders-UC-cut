"""Sparse search QUBO for block SA; energies are search scores, never bounds."""
# 模块职责：把块内硬约束行、跨块边界行、容量与可行性割编码为稀疏 QUBO，
# 先由硬约束传播固定位，再对自由位加权罚函数和费用代理。
from collections import deque
from itertools import product
import math
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .constraints import legacy_square_policy


# 用低位二进制权重加末位剩余量，覆盖整数区间 [0, maximum] 而不超过给定上界。
# 拒绝超过双精度精确整数范围的编码，maximum=0 时不需要辅助位。
def bit_weights(maximum):
    maximum = int(maximum)
    if not 0 <= maximum <= 2 ** 53:
        raise ValueError("Integer encoding exceeds exact double range")
    if not maximum:
        return np.array([], dtype=float)
    n = maximum.bit_length()
    return np.asarray([2 ** j for j in range(n - 1)] + [maximum - (2 ** (n - 1) - 1)], float)


def _integer_bounds(a, c):
    if not np.isfinite(c) or not np.isfinite(a).all():
        raise ValueError("Integer encoding exceeds exact double range")
    if abs(int(c)) + sum(abs(int(v)) for v in a) > 2 ** 53:
        raise ValueError("Integer encoding exceeds exact double range")
    return (int(c) + sum(int(v) for v in a if v < 0),
            int(c) + sum(int(v) for v in a if v > 0))


def encode_integer(value, weights):
    """Canonical representation for bit_weights, including its non-power-of-two last bit."""
    if not np.isfinite(value) or value != int(value) or not 0 <= value <= sum(weights):
        raise ValueError("Auxiliary value is outside its integer encoding")
    if not len(weights):
        return np.empty(0, dtype=np.uint8)
    value = int(value)
    last = int(value > 2 ** (len(weights)-1)-1)
    value -= last * int(weights[-1])
    return np.asarray([(value >> j) & 1 for j in range(len(weights)-1)] + [last], dtype=np.uint8)


def polish_auxiliaries(q, sample):
    """Minimize over every auxiliary bit, keeping all original free bits unchanged."""
    sample = np.asarray(sample)
    if sample.shape != (q.Q.shape[0],) or not np.isin(sample, [0, 1]).all():
        raise ValueError("Auxiliary correction requires a complete binary sample")
    x = sample.astype(np.uint8, copy=True)
    e = q.envelope
    tau = max(0., float((e.integer_alpha + e.integer_beta @ x[:len(q.free)]).max(initial=0.)))
    x[e.tau_ids] = encode_integer(tau, e.tau_weights)
    for row in q.slacks:
        target = -row.constant - row.coefficients @ x[row.variables]
        value = np.clip(np.rint(target), 0, row.maximum)
        x[row.ids] = encode_integer(value, row.weights)
    return x


def qubo_energy(q, sample):
    """Evaluate squared residuals before summing, avoiding cancellation in x.T Q x."""
    residual = q.penalty_matrix @ sample + q.penalty_constants
    return float(q.linear @ sample + q.linear_offset + sample @ q.direct @ sample + residual @ residual)


# 反复穷举至多四个变量的硬约束行（m 为 hard_rows 的结果），包含跨小时状态转移；只固定所有合法组合共有的值。
# 未确定值使用 NaN 标记；无满足组合说明硬约束不一致。
def fixed_binary(m):
    return _fixed_binary(m, m.A.tocsr())


def _fixed_binary(m, binary):
    fixed = np.full(binary.shape[1], np.nan)
    rows = []
    adjacent = [[] for _ in fixed]
    for k in range(binary.shape[0]):
        start, end = binary.indptr[k:k+2]
        if end-start <= 4:
            js, values = binary.indices[start:end], binary.data[start:end]
            for index in js:
                adjacent[index].append(len(rows))
            rows.append((k, js, values))
    pending = np.ones(len(rows), bool)
    queue = deque(range(len(rows)))
    while queue:
        row = queue.popleft()
        pending[row] = False
        k, js, values = rows[row]
        known = ~np.isnan(fixed[js])
        c = -m.rhs[k] + values[known] @ fixed[js[known]]
        free, a = js[~known], values[~known]
        if not len(free):
            if not (abs(c) <= 1e-9 if m.eq[k] else c <= 1e-9):
                raise ValueError("Inconsistent hard constraints")
            continue
        valid = []
        for bits in product((0, 1), repeat=len(free)):
            residual = c + a @ bits
            if (abs(residual) <= 1e-9 if m.eq[k] else residual <= 1e-9):
                valid.append(bits)
        if not valid:
            raise ValueError("Inconsistent hard constraints")
        for j, index in enumerate(free):
            if all(v[j] == valid[0][j] for v in valid):
                fixed[index] = valid[0][j]
                for neighbor in adjacent[index]:
                    if not pending[neighbor]:
                        queue.append(neighbor)
                        pending[neighbor] = True
    return fixed


# 搜索 QUBO 编码全天割投影的最大值；τ 与松弛位只服务搜索，不属于物理方案。
# 硬约束按行归一化，容量只用 u 列。能量包含罚项和量化误差，不能作为任何界。
# m 为块的 hard_rows 结果。
def search_qubo(m, b, cuts, point, boundary, o, *, constraint_policy=None, extra_rows=()):
    """Build a block QUBO using an injected constraint policy.

    The positional interface remains the archived one. Policies only change
    the capacity/feasibility rows and never turn QUBO energy into a certificate.
    """
    constraint_policy = constraint_policy or legacy_square_policy
    nx = m.nx
    if np.shape(point) != (nx,) or not np.isin(point, [0, 1]).all():
        raise ValueError("Search QUBO requires a complete binary cost-reference point")
    binary = m.A.tocsr()
    base_rows = [(binary.indices[binary.indptr[k]:binary.indptr[k + 1]],
                  binary.data[binary.indptr[k]:binary.indptr[k + 1]], m.rhs[k], m.eq[k], "hard")
                 for k in range(binary.shape[0])]
    base_rows += [(np.asarray(js, int), np.asarray(values, float), rhs, eq, "hard")
                  for js, values, rhs, eq in boundary]
    policy_rows = constraint_policy.rows(b, cuts, o, extra_rows)
    I = [k for k, row in enumerate(base_rows) for _ in row[0]]
    J = [j for row in base_rows for j in row[0]]
    V = [v for row in base_rows for v in row[1]]
    view = NS(rhs=np.asarray([row[2] for row in base_rows], float),
              eq=np.asarray([row[3] for row in base_rows], bool))
    fixed = _fixed_binary(view, sparse.coo_matrix((V, (I, J)), shape=(len(base_rows), nx)).tocsr())
    rows = base_rows + [(row.indices, row.values, row.rhs, row.equality, row.kind)
                        for row in policy_rows]
    free = np.flatnonzero(np.isnan(fixed))
    mapping = np.full(nx, -1); mapping[free] = np.arange(len(free))
    qi, qj, qv, R, constants = [], [], [], [], []
    nv = len(free)
    offset = 0.
    slacks, feasibility_bounds = [], []
    local_bound = max(1., float(np.abs(b.obj[b.binary][free]).sum(initial=0.)))
    for cut in cuts:
        local_bound += float(np.abs(cut.beta[free]).sum())

    # 加入 weight*(a*x+c)^2 或 weight*max(0, a*x+c)^2；系数先按整行最大值归一化。
    # 两自由位以内用精确真值表；可行性割的整数化只放松搜索罚项，不误罚真实可行点。
    def penalty(js, values, rhs, equality, weight, step, kind='hard'):
        nonlocal nv, offset
        js, values = np.asarray(js, int), np.asarray(values, float)
        scale = np.abs(values).max(initial=0.)
        if not scale:
            return
        values, c, step = values / scale, -rhs / scale, step / scale
        known = ~np.isnan(fixed[js])
        c += values[known] @ fixed[js[known]]
        ids, a = mapping[js[~known]], values[~known]
        if not len(ids):
            offset += weight * (c if equality else max(0., c)) ** 2
            return
        if not equality:
            low, high = c + np.minimum(a, 0).sum(), c + np.maximum(a, 0).sum()
            if high <= 1e-9:
                return
            if len(ids) <= 2:
                e00 = max(0, c) ** 2
                offset += weight * e00
                e10 = max(0, c + a[0]) ** 2
                qi.append(ids[0]); qj.append(ids[0]); qv.append(weight * (e10 - e00))
                if len(ids) == 2:
                    e01, e11 = max(0, c + a[1]) ** 2, max(0, c + a.sum()) ** 2
                    mixed = weight * (e11 - e10 - e01 + e00) / 2
                    qi.extend([ids[1], ids[0], ids[1]]); qj.extend([ids[1], ids[1], ids[0]])
                    qv.extend([weight * (e01 - e00), mixed, mixed])
                return
            if kind == 'feasibility':
                step = 1. / (64 * (len(ids)+1))
                integer_a, integer_c = np.floor(a / step), np.floor(c / step)
                low_integer, high_integer = _integer_bounds(integer_a, integer_c)
                feasibility_bounds.append((len(ids)+1) * step)
                if high_integer <= 0:
                    return
                a, c, low = step * integer_a, step * integer_c, step * low_integer
            maximum = max(0, math.floor(-low / step + 1e-9))
            weights = bit_weights(maximum)
            auxiliary = np.arange(nv, nv + len(weights))
            slacks.append(NS(ids=auxiliary, weights=weights, maximum=maximum, variables=ids,
                             coefficients=a / step, constant=c / step, step=step, weight=weight, kind=kind))
            ids, a = np.r_[ids, auxiliary], np.r_[a, step * weights]
            nv += len(weights)
        R.append((ids, np.sqrt(weight) * a)); constants.append(np.sqrt(weight) * c)

    for js, values, rhs, equality, kind in rows:
        if kind == "capacity":
            _integer_bounds(np.asarray(values, dtype=int), int(-rhs))
        penalty(js, values, rhs, equality, constraint_policy.weight(kind, local_bound, o), 1., kind)

    # 基线割加非负修正 τ，避免把多条费用斜率相加；固定变量先代入截距。
    optimality = [c for c in cuts if c.optimality]
    cost = b.obj[b.binary].copy()
    base = max(optimality, key=lambda c: c.alpha + c.beta @ point) if optimality else None
    known = np.flatnonzero(~np.isnan(fixed))
    differences, intercepts = [], []
    if optimality:
        cost += base.beta
        for cut in optimality:
            if cut is base:
                continue
            beta = cut.beta - base.beta
            differences.append(beta[free])
            intercepts.append(cut.alpha - base.alpha + beta[known] @ fixed[known])
    differences = np.asarray(differences).reshape(len(intercepts), len(free))
    unit = max(float(np.abs(cost[free]).max(initial=0.)),
               float(np.abs(differences).max(initial=0.)), 1e-9)
    differences, intercepts = differences / unit, np.asarray(intercepts) / unit
    delta = 1. / (64 * (len(free) + 1))
    integer_beta, integer_alpha = np.rint(differences / delta), np.rint(intercepts / delta)
    lows, highs = [], []
    for a, c in zip(integer_beta, integer_alpha):
        low, high = _integer_bounds(a, c)
        lows.append(low); highs.append(high)
    maximum = max([0, *highs])
    tau_weights = bit_weights(maximum)
    tau_ids = np.arange(nv, nv + len(tau_weights)); nv += len(tau_weights)
    slack_bits = 0
    for a, c, low, high in zip(integer_beta, integer_alpha, lows, highs):
        if high <= 0:
            continue
        slack = bit_weights(maximum - low)
        variables = np.r_[np.arange(len(free)), tau_ids]
        auxiliary = np.arange(nv, nv + len(slack))
        slacks.append(NS(ids=auxiliary, weights=slack, maximum=maximum-low, variables=variables,
                         coefficients=np.r_[a, -tau_weights], constant=c, step=1., weight=2*delta, kind='optimality'))
        ids = np.r_[variables, auxiliary]
        # 整数违约至少为 1；2Δ*k² > Δ*k，故固定原始位后最优辅助位不会低估量化包络。
        R.append((ids, np.sqrt(2 * delta) * np.r_[a, -tau_weights, slack]))
        constants.append(np.sqrt(2 * delta) * c)
        nv += len(slack); slack_bits += len(slack)
    offset += ((base.alpha if base is not None else 0.) + cost[known] @ fixed[known]) / unit
    linear = np.zeros(nv)
    linear[:len(free)] = cost[free] / unit
    linear[tau_ids] = delta * tau_weights
    I = [k for k, (ids, _) in enumerate(R) for _ in ids]
    J = [j for ids, _ in R for j in ids]
    V = [v for _, values in R for v in values]
    A = sparse.coo_matrix((V, (I, J)), shape=(len(R), nv)).tocsr()
    c = np.asarray(constants)
    direct = sparse.coo_matrix((qv, (qi, qj)), shape=(nv, nv)).tocsr()
    Q = direct + A.T @ A + sparse.diags(2 * (A.T @ c) + linear)
    Q.eliminate_zeros()
    if not np.isfinite(Q.data).all():
        raise ValueError("Nonfinite search QUBO coefficients")
    envelope = NS(beta=differences, alpha=intercepts, integer_beta=integer_beta,
                  integer_alpha=integer_alpha, delta=delta, tau_ids=tau_ids, tau_weights=tau_weights,
                  slack_bits=slack_bits, cuts=int(bool(optimality)) + sum(high > 0 for high in highs),
                  error_bound=(len(free) + 1) * delta / 2 if len(intercepts) else 0.)
    return NS(Q=Q.tocsr(), cost=cost, cost_unit=unit, offset=offset + float(c @ c), envelope=envelope,
              slacks=slacks, penalty_matrix=A, penalty_constants=c, direct=direct,
              linear=linear, linear_offset=offset, feasibility_relaxation_bound=max(feasibility_bounds, default=0.),
              fixed=fixed, free=free, nx=nx, energy_unit=1.,
              constraint_policy=constraint_policy.name, constraint_rows=len(policy_rows),
              local_objective_bound=local_bound)
