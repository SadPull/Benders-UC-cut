"""Commitment repair, startup fees and independent physical/cost validation."""
# 模块职责：转换 u/v/w 与模型向量、进行启停/容量修复，核算实际冷热启动费用，并独立校验全天物理方程与费用。
# 修复只提供候选；通过连续调度及独立 validate 后，方案费用才可更新原问题上界。
from types import SimpleNamespace as NS
import hashlib

import numpy as np


def transition_allowed(d):
    """Necessary startup/shutdown reachability under the unchanged ramp model."""
    start = d.pmin <= np.minimum(d.pstart, d.ramp_up) + 1e-7
    stop = np.repeat((d.pmin <= np.minimum(d.pshut, d.ramp_down) + 1e-7)[:, None], d.horizon, axis=1)
    stop[:, 0] = d.initial_output <= d.ramp_down + 1e-7
    return start, stop


# 按初始历史修复启停时序、必要的启停爬坡条件和场景容量；连续可行性仍由调度判断。
# 返回重建 u/v/w 或 None；容量修复可把机组从允许最早开机时刻一直打开，因此会修改其他小时。
def repair(d, scenarios, u):
    u = np.asarray(u)
    if u.shape != (d.n_units, d.horizon) or not np.isfinite(u).all():
        return None
    u = np.clip(np.rint(u), 0, 1)
    start_allowed, stop_allowed = transition_allowed(d)
    def timeline(i, row):
        row = row.copy()
        state, duration = d.initial_on[i], d.initial_duration[i]
        for t in range(d.horizon):
            minimum = d.min_up[i] if state else d.min_down[i]
            target = row[t]
            if target != state:
                blocked = duration < minimum or (t == d.horizon - 1 and state == 1)
                if state == 0:
                    blocked |= not start_allowed[i]
                else:
                    blocked |= not stop_allowed[i, t]
                if blocked:
                    target = state
            duration = duration + 1 if target == state else 1
            state = target; row[t] = state
        return row

    for i in range(d.n_units):
        u[i] = timeline(i, u[i])
    loads = np.asarray([s.load for s in scenarios])
    required = (1 + d.reserve_fraction) * loads.max(axis=0)
    minimum_load = loads.min(axis=0)
    if (np.sum(d.pmin[:, None] * u, axis=0) > minimum_load + 1e-6).any():
        return None

    # 重建当前试探方案的 v/w，返回打包向量及每小时最坏场景负荷加备用的容量缺口。
    def evaluate(on):
        x = pack(d, on)
        n = d.n_units * d.horizon
        v, w = x[n:2 * n].reshape(on.shape, order="F"), x[2 * n:].reshape(on.shape, order="F")
        return x, np.maximum(0, required - capacity(d, on, v, w).sum(axis=0))

    # 每轮只选择能减少首个容量缺口且不扩大其他缺口的开机扩展；不能修复则返回 None。
    for _ in range(d.n_units + 1):
        candidate, deficit = evaluate(u)
        if (deficit <= 1e-6).all():
            return candidate
        t = np.flatnonzero(deficit > 1e-6)[0]
        choices = []
        for i in range(d.n_units):
            start = 0 if d.initial_on[i] else max(0, int(d.min_down[i] - d.initial_duration[i]))
            if start >= d.horizon:
                continue
            trial = u.copy(); trial[i, start:] = 1
            trial[i] = timeline(i, trial[i])
            if (np.sum(d.pmin[:, None] * trial, axis=0) > minimum_load + 1e-6).any():
                continue
            x, remaining = evaluate(trial)
            gain, cost = deficit[t] - remaining[t], fixed_cost(d, trial, commitment(d, x).w)
            if (remaining <= deficit + 1e-6).all() and gain > 1e-6:
                choices.append((gain, -cost, -i, trial))
        if not choices:
            return None
        u = max(choices, key=lambda item: item[:3])[3]
    return None


# 不用稀疏模型行标签，直接重算二进制时序、初始/终端规则、出力、备用、爬坡和费用。
# 失败抛 ValueError；成功返回物理最大残差与总费用残差。
def validate(d, scenarios, result):
    if not np.isfinite(result.objective):
        raise ValueError("No finite objective")
    for name in ("u", "v", "w"):
        z = getattr(result, name)
        if z.shape != (d.n_units, d.horizon) or not np.isin(z, [0, 1]).all():
            raise ValueError("Invalid binary schedule")
    u, v, w = result.u, result.v, result.w
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    if not np.array_equal(v, np.maximum(u - previous, 0)) or not np.array_equal(w, np.maximum(previous - u, 0)):
        raise ValueError("Startup/shutdown transition mismatch")
    if w[:, -1].any():
        raise ValueError("Terminal shutdown is forbidden")
    for i in range(d.n_units):
        state, duration = d.initial_on[i], d.initial_duration[i]
        for t in range(d.horizon):
            minimum = d.min_up[i] if state else d.min_down[i]
            if u[i, t] != state:
                if duration < minimum:
                    raise ValueError("Minimum up/down time violated")
                state, duration = u[i, t], 1
            else:
                duration += 1
    if len(result.scenarios) != len(scenarios):
        raise ValueError("Missing scenarios")
    maximum, fuel = 0., 0.
    for s, z in zip(scenarios, result.scenarios):
        p, r = z.p, z.r
        if (z.name != s.name or p.shape != u.shape or r.shape != u.shape
                or not np.isfinite(p).all() or not np.isfinite(r).all()):
            raise ValueError("Invalid dispatch")
        previous = np.column_stack([d.initial_output, p[:, :-1]])
        residuals = [np.abs(p.sum(axis=0) - s.load), d.reserve_fraction * s.load - r.sum(axis=0),
                     d.pmin[:, None] * u - p, p + r - capacity(d, u, v, w),
                     p + r - d.pmax[:, None] * u, p - previous - d.ramp_up[:, None],
                     previous - p - d.ramp_down[:, None], -p, -r]
        maximum = max(maximum, *(float(a.max()) for a in residuals))
        cost = float(np.sum(d.cost_a[:, None] * p ** 2 + d.cost_b[:, None] * p))
        if abs(cost - z.objective) > 1e-4:
            raise ValueError("Scenario fuel cost mismatch")
        fuel += s.probability * cost
    fixed = fixed_cost(d, u, w)
    # 冷启动差额已包含在实际 fixed 中；总费用只能再加概率加权的纯燃料，不能重复加小时补偿。
    error = float(abs(fixed + fuel - result.objective))
    if maximum > 1e-5 or error > 1e-4:
        raise ValueError("Physical or total cost validation failed")
    return {"feasible": True, "max_residual_mw": maximum, "cost_residual": error}


# 从全天 u 和初始状态重建启动 v、停机 w，并按每类变量的列主序拼成二进制向量。
def pack(d, u):
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    v, w = np.maximum(u - previous, 0), np.maximum(previous - u, 0)
    return np.concatenate([z.ravel(order="F") for z in (u, v, w)])


# 将 u/v/w 向量按变量类型分段，恢复为机组 × 小时矩阵并复制，避免修改原缓存。
def commitment(d, x):
    # Each variable type occupies a contiguous, column-major unit/hour matrix.
    n = d.n_units * d.horizon
    return NS(**{name: np.asarray(x)[j*n:(j+1)*n].reshape((d.n_units, d.horizon), order="F").copy()
                 for j, name in enumerate(("u", "v", "w"))})


# 对完整 u/v/w 的二进制字节生成 SHA256，作为全天方案去重与反馈关联身份。
def commitment_hash(z):
    return hashlib.sha256(np.r_[z.u.ravel(), z.v.ravel(), z.w.ravel()].astype(np.uint8).tobytes()).hexdigest()


# 逐机组逐小时计算启动容量与下一小时停机容量的较小值，供备用检查及独立物理验证使用。
def capacity(d, u, v, w):
    startup = d.pmax[:, None] * u - (d.pmax - d.pstart)[:, None] * v
    shutdown = d.pmax[:, None] * u - (d.pmax - d.pshut)[:, None] * np.column_stack([w[:, 1:], np.zeros(d.n_units)])
    return np.minimum(startup, shutdown)


# 从完整模型点恢复 u/v/w 和分场景 p/r，返回原目标费用及未乘概率的场景燃料明细。
def unpack(m, point):
    d = m.d
    result = NS(objective=float(m.obj @ point + m.q @ (point ** 2)), scenarios=[])
    for name in ("u", "v", "w"):
        setattr(result, name, point[getattr(m.ix, name)].copy())
    for k, s in enumerate(m.scenarios):
        p = point[m.ix.p[:, :, k]]
        r = point[m.ix.r[:, :, k]]
        result.scenarios.append(NS(name=s.name, probability=s.probability, p=p.copy(), r=r.copy(),
                                   objective=float(np.sum(d.cost_a[:, None] * p ** 2 + d.cost_b[:, None] * p))))
    return result


# 返回全天空载运行费用 + 实际启动费用 + 停机费用，不含任何场景燃料或不平衡罚费用。
def fixed_cost(d, u, w):
    return float(np.sum(d.cost_c[:, None]*u + startup_costs(d, u) + d.shutdown_cost[:, None]*w))


# 从 u 和初始开机状态重建启动时刻，返回每台机组每小时的实际热/冷启动费用。
# 只有启动前连续停机小时数严格大于 min_down+cold_start_hours 才使用冷启动费。
def startup_costs(d, u):
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    starts = np.maximum(u-previous, 0)
    costs = d.startup_cost[:, None]*starts
    if not hasattr(d, "cold_startup_cost"):
        return costs
    offline = np.where(d.initial_on, 0, d.initial_duration).copy()
    threshold = d.min_down+d.cold_start_hours
    for t in range(d.horizon):
        # 先按启动前的停机时长定价，再用本小时开停机状态更新计数，避免差一小时。
        cold = (starts[:, t] > 0) & (offline > threshold)
        costs[cold, t] = d.cold_startup_cost[cold]
        offline = np.where(u[:, t] > 0, 0, offline+1)
    return costs
