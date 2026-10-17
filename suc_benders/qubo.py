"""Sparse assembly of archived penalties and whole-cut quantization budgets."""
# 模块职责：将离散 Benders 主问题编码为稀疏 QUBO，处理固定变量、theta、罚函数和松弛辅助位。
# 主问题末列是费用 theta；容量用有理数格点，Benders 割使用整条误差预算向下量化。
from fractions import Fraction
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


# 将整条割误差预算均分到非零系数和常数，并全部向下取整，返回整数系数、常数与步长。
# exact 系数必须精确落在格点上；theta 位系数不能因独立取整破坏原目标步长。
def quantize(coefficients, constant, budget, exact=None):
    coefficients = {j: v for j, v in coefficients.items() if v}
    step = Fraction(str(budget)) / (len(coefficients) + 1)
    values = {j: math.floor(Fraction(float(v)) / step) for j, v in coefficients.items()}
    c = math.floor(Fraction(float(constant)) / step)
    for j, v in (exact or {}).items():
        integer = v / step
        if integer.denominator != 1:
            raise ValueError("Exact objective coefficient is not on the quantization grid")
        values[j] = int(integer)
    return values, c, step


# 反复穷举至多四个变量的 hard 行，包含跨小时状态转移；只固定所有合法组合共有的值。
# 未确定值使用 NaN 标记；无满足组合说明硬约束不一致。
def fixed_binary(m):
    fixed = np.full(len(m.binary), np.nan)
    rows = []
    for k, label in enumerate(m.kind):
        row = m.A[k, :len(fixed)].tocsr()
        if label == "hard" and row.nnz <= 4:
            rows.append((k, row.indices, row.data))
    changed = True
    while changed:
        changed = False
        for k, js, values in rows:
            known = ~np.isnan(fixed[js])
            c = -m.rhs[k] + values[known] @ fixed[js[known]]
            free, a = js[~known], values[~known]
            valid = []
            for bits in product((0, 1), repeat=len(free)):
                residual = c + a @ bits
                if (abs(residual) <= 1e-9 if m.eq[k] else residual <= 1e-9):
                    valid.append(bits)
            if not valid:
                raise ValueError("Inconsistent hard constraints")
            for j, index in enumerate(free):
                if all(v[j] == valid[0][j] for v in valid):
                    fixed[index] = valid[0][j]; changed = True
    return fixed


# 输入线性主问题、编码参数与块费用上界，返回 Q、常数偏移、固定/自由位及 theta 恢复信息。
# 不等式用非负松弛位转成平方罚项；AND 使用原 Rosenberg 罚函数，输出能量只供搜索。
def qubo(m, o, upper):
    nb = len(m.binary)
    fixed = fixed_binary(m)
    free, known = np.flatnonzero(np.isnan(fixed)), np.flatnonzero(~np.isnan(fixed))
    mapping = dict(zip(free, range(len(free))))
    lower = math.floor(m.lb[-1] / o.theta_step) * o.theta_step
    upper = math.ceil(min(upper, m.ub[-1]) / o.theta_step) * o.theta_step
    if not np.isfinite(upper) or upper < lower:
        raise ValueError("Invalid epigraph range")
    integer_weights = bit_weights(round((upper - lower) / o.theta_step))
    weights = o.theta_step * integer_weights
    theta_ids = np.arange(len(free), len(free) + len(weights))
    nv = len(free) + len(weights)
    offset = lower / m.scale
    qi, qj, qv = list(theta_ids), list(theta_ids), list(weights / m.scale)
    penalty_rows, constants = [], []

    # 向 QUBO 添加带 penalty 的二次乘积；代入已固定值，并将交叉系数对称分配给 Q[i,j]/Q[j,i]。
    def add_product(i, j, value):
        nonlocal offset
        value *= o.penalty
        if not np.isnan(fixed[i]):
            value *= fixed[i]
            if not np.isnan(fixed[j]):
                offset += value * fixed[j]
                return
            i = j
        elif not np.isnan(fixed[j]):
            value *= fixed[j]; j = i
        i, j = mapping[i], mapping[j]
        qi.extend([i, j]); qj.extend([j, i]); qv.extend([value / 2, value / 2])

    # AND 关系 z=v*w 使用 v*w-2*v*z-2*w*z+3*z，避免重复使用三条平方罚项改变能量尺度。
    # Archived Rosenberg AND penalty, not three squared inequalities.
    for z, v, w in m.and_pairs:
        for i, j, coefficient in ((v, w, 1), (v, z, -2), (w, z, -2), (z, z, 3)):
            add_product(i, j, coefficient)

    removed_rows = []
    for k, label in enumerate(m.kind):
        if label == "and":
            continue
        row = m.A[k, :nb].tocsr()
        a = {mapping[j]: float(v) for j, v in zip(row.indices, row.data) if j in mapping and v}
        c = float((m.A[k, known] @ fixed[known]).item() - m.rhs[k] + m.A[k, nb] * lower)
        if label == "hard" and not m.eq[k] and len(a) <= 2:
            ids = list(a); values = list(a.values())
            e00 = max(0, c) ** 2; offset += o.penalty * e00
            if ids:
                e10 = max(0, c + values[0]) ** 2
                qi.append(ids[0]); qj.append(ids[0]); qv.append(o.penalty * (e10 - e00))
                if len(ids) == 2:
                    e01 = max(0, c + values[1]) ** 2
                    e11 = max(0, c + sum(values)) ** 2
                    mixed = o.penalty * (e11 - e10 - e01 + e00) / 2
                    qi.extend([ids[1], ids[0], ids[1]]); qj.extend([ids[1], ids[1], ids[0]])
                    qv.extend([o.penalty * (e01 - e00), mixed, mixed])
            continue
        if label == "hard":
            scale = 1.
        elif label == "capacity":
            # 分母最小公倍数和系数最大公约数确定精确容量格点，不使用浮点四舍五入替代。
            original, constant = getattr(m, "capacity_rows", {}).get(k, (
                {j: Fraction(str(v)) for j, v in zip(row.indices, row.data)}, Fraction(str(-m.rhs[k]))))
            constant += sum(v * int(fixed[j]) for j, v in original.items() if not np.isnan(fixed[j]))
            exact = {mapping[j]: v for j, v in original.items() if j in mapping and v}
            denominator = math.lcm(*(v.denominator for v in exact.values()))
            integers = {j: int(v * denominator) for j, v in exact.items()}
            divisor = math.gcd(*integers.values()) or 1
            step = Fraction(divisor, denominator)
            a, c = {j: v // divisor for j, v in integers.items()}, math.ceil(constant / step)
            scale = float(step / Fraction(str(o.reserve_scale)))
        else:
            budget, norm = (o.theta_step, m.scale) if label == "optimality" else (1e-3, o.reserve_scale)
            exact = {j: Fraction(str(float(m.A[k, nb]))) * Fraction(str(o.theta_step)) * int(w)
                     for j, w in zip(theta_ids, integer_weights)} if label == "optimality" else None
            a, c, step = quantize(a, c, budget, exact)
            scale = float(step / Fraction(str(norm)))
        low = c + sum(min(0, v) for v in a.values())
        high = c + sum(max(0, v) for v in a.values())
        if not m.eq[k] and high <= 0:
            removed_rows.append(k)
            continue
        sw = bit_weights(max(0, -low)) if not m.eq[k] else np.array([])
        # 对 a*x+c<=0 添加非负松弛，变成 a*x+c+slack=0；等式行不需要松弛位。
        ids = np.arange(nv, nv + len(sw)); nv += len(sw)
        indices, values = np.r_[list(a), ids].astype(int), np.r_[list(a.values()), sw] * scale
        nonzero = values != 0
        penalty_rows.append((indices[nonzero], values[nonzero])); constants.append(c * scale)
    I, J, V = [], [], []
    for i, (indices, values) in enumerate(penalty_rows):
        I.extend([i] * len(indices)); J.extend(indices); V.extend(values)
    R = sparse.coo_matrix((V, (I, J)), shape=(len(constants), nv)).tocsr()
    c = np.asarray(constants)
    Q = sparse.coo_matrix((qv, (qi, qj)), shape=(nv, nv)).tocsr()
    # 展开 penalty*||R*x+c||^2；二进制 x_i^2=x_i，因此线性项写到 Q 的对角线上。
    Q = Q + o.penalty * (R.T @ R + sparse.diags(2 * (R.T @ c)))
    Q.eliminate_zeros()
    offset += o.penalty * (c @ c)
    return NS(Q=Q.tocsr(), offset=float(offset), fixed=fixed, free=free, nx=m.nx,
              theta_ids=theta_ids, theta_weights=weights, theta_lower=lower, energy_unit=o.theta_step / m.scale,
              removed_rows=removed_rows)
