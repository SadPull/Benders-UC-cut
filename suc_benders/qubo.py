"""Sparse search QUBO for block SA; energies are search scores, never bounds."""
# 模块职责：把块内硬约束行、跨块边界行、容量与可行性割编码为稀疏 QUBO，
# 先由硬约束传播固定位，再对自由位加权罚函数和费用代理。
from fractions import Fraction
from collections import deque
from itertools import product
import math
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse


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


# 搜索专用 QUBO：只含 u/v/w 和容量/可行性割松弛位，能量单位为一个归一化成本单位。
# 成本取当前方案处最紧最优性割的线性代理；硬约束按行最大系数归一化后统一加权，容量只用 u 列，
# 避免 v/w 充当容量补偿位。能量只用于搜索，不能作为任何界。
# m 为块的 hard_rows 结果。
def search_qubo(m, b, cuts, point, boundary, o):
    nx = m.nx
    if np.shape(point) != (nx,) or not np.isin(point, [0, 1]).all():
        raise ValueError("Search QUBO requires a complete binary cost-reference point")
    binary = m.A.tocsr()
    rows = [(binary.indices[binary.indptr[k]:binary.indptr[k + 1]],
             binary.data[binary.indptr[k]:binary.indptr[k + 1]], m.rhs[k], m.eq[k]) for k in range(binary.shape[0])]
    rows += [(np.asarray(js, int), np.asarray(values, float), rhs, eq) for js, values, rhs, eq in boundary]
    I = [k for k, row in enumerate(rows) for _ in row[0]]
    J = [j for row in rows for j in row[0]]
    V = [v for row in rows for v in row[1]]
    view = NS(rhs=np.asarray([row[2] for row in rows], float), eq=np.asarray([row[3] for row in rows], bool))
    fixed = _fixed_binary(view, sparse.coo_matrix((V, (I, J)), shape=(len(rows), nx)).tocsr())
    free = np.flatnonzero(np.isnan(fixed))
    mapping = np.full(nx, -1); mapping[free] = np.arange(len(free))
    qi, qj, qv, R, constants = [], [], [], [], []
    nv = len(free)

    # 加入 weight*(a*x+c)^2 或 weight*max(0, a*x+c)^2；系数先按整行最大值归一化。
    # 至多两个自由位的不等式用精确真值表，其余不等式在 step 格点上用非负松弛位。
    def penalty(js, values, rhs, equality, weight, step):
        nonlocal nv
        js, values = np.asarray(js, int), np.asarray(values, float)
        scale = np.abs(values).max(initial=0.)
        if not scale:
            return
        values, c, step = values / scale, -rhs / scale, step / scale
        known = ~np.isnan(fixed[js])
        c += values[known] @ fixed[js[known]]
        ids, a = mapping[js[~known]], values[~known]
        if not len(ids):
            return
        if not equality:
            low, high = c + np.minimum(a, 0).sum(), c + np.maximum(a, 0).sum()
            if high <= 1e-9:
                return
            if len(ids) <= 2:
                e00 = max(0, c) ** 2
                e10 = max(0, c + a[0]) ** 2
                qi.append(ids[0]); qj.append(ids[0]); qv.append(weight * (e10 - e00))
                if len(ids) == 2:
                    e01, e11 = max(0, c + a[1]) ** 2, max(0, c + a.sum()) ** 2
                    mixed = weight * (e11 - e10 - e01 + e00) / 2
                    qi.extend([ids[1], ids[0], ids[1]]); qj.extend([ids[1], ids[1], ids[0]])
                    qv.extend([weight * (e01 - e00), mixed, mixed])
                return
            slack = step * bit_weights(max(0, math.floor(-low / step + 1e-9)))
            ids, a = np.r_[ids, np.arange(nv, nv + len(slack))], np.r_[a, slack]
            nv += len(slack)
        R.append((ids, np.sqrt(weight) * a)); constants.append(np.sqrt(weight) * c)

    for js, values, rhs, equality in rows:
        penalty(js, values, rhs, equality, o.sa_hard_weight, 1.)
    position = np.full(len(b.obj), -1); position[b.binary] = np.arange(nx)
    units = position[b.u]
    # 容量行只保留 pmax*u；启停降额由全天修复与独立校验复核。格点取各 pmax 的最大公约数。
    pmax = [Fraction(str(v)) for v in b.d.pmax]
    denominator = math.lcm(*(v.denominator for v in pmax))
    step = float(Fraction(math.gcd(*(int(v * denominator) for v in pmax)), denominator))
    for t, hour in enumerate(b.times):
        required = (1 + b.d.reserve_fraction) * max(s.load[hour] for s in b.scenarios)
        penalty(units[:, t], -b.d.pmax, -required, False, o.sa_capacity_weight, step)
    for c in cuts:
        if not c.optimality:
            penalty(np.arange(nx), c.beta, -c.alpha, False, o.sa_capacity_weight,
                    np.abs(c.beta).max(initial=1.) / 64)

    # 当前方案仅用于选择费用代理，不添加邻近惩罚。
    optimality = [c for c in cuts if c.optimality]
    cost = b.obj[b.binary].copy()
    if optimality:
        cost += max(optimality, key=lambda c: c.alpha + c.beta @ point).beta
    unit = max(float(np.abs(cost[free]).max(initial=0.)), 1e-9)
    linear = np.zeros(nv)
    linear[:len(free)] = cost[free] / unit
    I = [k for k, (ids, _) in enumerate(R) for _ in ids]
    J = [j for ids, _ in R for j in ids]
    V = [v for _, values in R for v in values]
    A = sparse.coo_matrix((V, (I, J)), shape=(len(R), nv)).tocsr()
    c = np.asarray(constants)
    Q = sparse.coo_matrix((qv, (qi, qj)), shape=(nv, nv)).tocsr()
    Q = Q + A.T @ A + sparse.diags(2 * (A.T @ c) + linear)
    Q.eliminate_zeros()
    return NS(Q=Q.tocsr(), cost=cost, cost_unit=unit,
              fixed=fixed, free=free, nx=nx, energy_unit=1.)
