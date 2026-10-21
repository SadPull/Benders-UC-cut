"""Sparse SUC model; all rows are A @ point <= rhs, or equality rows.

Variable ordering matches MATLAB: variable type, scenario, hour, unit.
Internal indices and hours are zero based; schedule matrices are unit x hour.
"""
# 模块职责：建立标准随机 UC 稀疏模型、计算模型指纹，并按跨块拉格朗日乘子分解时间块。
# u/v/w 为开机/启动/停机，p/r 为场景出力/备用；矩阵索引从 0 开始，方案形状为机组 × 小时。
from types import SimpleNamespace as NS
import hashlib
import json

import numpy as np
from scipy import sparse


# 返回 A、rhs、eq、线性/二次目标、边界及变量索引。
# u/v/w 费用跨场景只计一次，p 的燃料系数乘场景概率；正冷启动差额另用连续变量承载。
def model(d, scenarios):
    n, T, S = d.n_units, d.horizon, len(scenarios)
    N = n * T
    # 列主序使机组索引变化最快，其次小时、场景；与 pack/unpack 及历史 MATLAB 索引一致。
    u = np.arange(N).reshape((n, T), order="F")
    p = np.arange(3 * N, (3 + S) * N).reshape((n, T, S), order="F")
    ix = NS(u=u, v=u + N, w=u + 2 * N, p=p, r=p + N * S)
    nv = (3 + 2 * S) * N
    # 热启动基费在 v 中；冷启动仅通过共享连续列补足差额，因此不会被场景概率重复收费。
    cold = hasattr(d, "cold_startup_cost") and (d.cold_startup_cost > d.startup_cost).any()
    if cold:
        ix.cold_start = np.arange(nv, nv+N).reshape((n, T), order="F")
        nv += N
    m = NS(obj=np.zeros(nv), q=np.zeros(nv), lb=np.zeros(nv), ub=np.ones(nv), ix=ix,
           binary=np.arange(3 * N), time=np.tile(np.repeat(np.arange(T), n), 3 + 2 * S + int(cold)),
           d=d, scenarios=scenarios, constant=0.)
    m.obj[ix.u], m.obj[ix.v], m.obj[ix.w] = d.cost_c[:, None], d.startup_cost[:, None], d.shutdown_cost[:, None]
    if cold:
        m.obj[ix.cold_start] = (d.cold_startup_cost-d.startup_cost)[:, None]
    for k, s in enumerate(scenarios):
        m.obj[ix.p[:, :, k]] = s.probability * d.cost_b[:, None]
        m.q[ix.p[:, :, k]] = s.probability * d.cost_a[:, None]
        m.ub[ix.p[:, :, k]] = d.pmax[:, None]
        m.ub[ix.r[:, :, k]] = d.pmax[:, None]
    I, J, V, rhs, eq, labels = [], [], [], [], [], []

    # 追加一条 A @ point <= rhs 或等式约束，忽略零系数，并保留用于审计的物理标签。
    def add(js, values, bound, equality=False, label=""):
        js, values = np.atleast_1d(js), np.atleast_1d(values)
        keep = values != 0
        I.extend([len(rhs)] * int(keep.sum()))
        J.extend(js[keep]); V.extend(values[keep])
        rhs.append(bound); eq.append(equality); labels.append(label)

    for t in range(T):
        for i in range(n):
            u, v, w = ix.u[i, t], ix.v[i, t], ix.w[i, t]
            if t == 0:
                add([u, v, w], [1, -1, 1], d.initial_on[i], True, "transition")
            else:
                add([u, v, w, ix.u[i, t - 1]], [1, -1, 1, -1], 0, True, "transition")
            add([v, w], [1, 1], 1, label="exclusive")
            if t == T - 1:
                add(w, 1, 0, True, "terminal")
            remaining = d.initial_on[i] * d.min_up[i] + (1 - d.initial_on[i]) * d.min_down[i] - d.initial_duration[i]
            if t < remaining:
                add(u, 1, d.initial_on[i], True, "initial")
            if cold:
                # Cold iff the preceding L+1 hours are all OFF, including pre-horizon history.
                window = int(d.min_down[i]+d.cold_start_hours[i])+1
                history = ix.u[i, max(0, t-window):t]
                initially_warm = t < window and (d.initial_on[i] or d.initial_duration[i]+t < window)
                add([v, ix.cold_start[i, t], *history], [1, -1, *-np.ones(len(history))],
                    float(initially_warm), label="cold_start")
            # 最小开停机窗口截断至优化时域末端，不额外假设时域之外的启停状态。
            for j in range(t, min(T, t + int(d.min_up[i]))):
                add([v, ix.u[i, j]], [1, -1], 0, label="min_up")
            for j in range(t, min(T, t + int(d.min_down[i]))):
                add([w, ix.u[i, j]], [1, 1], 1, label="min_down")
            for k in range(S):
                p, r = ix.p[i, t, k], ix.r[i, t, k]
                add([u, p], [d.pmin[i], -1], 0, label="pmin")
                add([p, r, u, v], [1, 1, -d.pmax[i], d.pmax[i] - d.pstart[i]], 0, label="startup_capacity")
                add([p, r, u], [1, 1, -d.pmax[i]], 0, label="envelope")
                if t < T - 1:
                    add([p, r, u, ix.w[i, t + 1]], [1, 1, -d.pmax[i], d.pmax[i] - d.pshut[i]], 0, label="shutdown_capacity")
                else:
                    add([p, r, u], [1, 1, -d.pmax[i]], 0, label="shutdown_capacity")
                if t == 0:
                    add(p, 1, d.ramp_up[i] + d.initial_output[i], label="ramp_up")
                    add(p, -1, d.ramp_down[i] - d.initial_output[i], label="ramp_down")
                else:
                    add([p, ix.p[i, t - 1, k]], [1, -1], d.ramp_up[i], label="ramp_up")
                    add([p, ix.p[i, t - 1, k]], [-1, 1], d.ramp_down[i], label="ramp_down")
        for k, s in enumerate(scenarios):
            add(ix.p[:, t, k], np.ones(n), s.load[t], True, "balance")
            add(ix.r[:, t, k], -np.ones(n), -d.reserve_fraction * s.load[t], label="reserve")
    m.A = sparse.coo_matrix((V, (I, J)), shape=(len(rhs), nv)).tocsr()
    m.rhs, m.eq, m.labels = np.asarray(rhs, float), np.asarray(eq, bool), labels
    return m


# 按固定字节序散列完整稀疏约束、目标、边界、变量身份和场景信息，生成恢复记录的模型身份。
# 排序 CSR 索引以消除存储顺序影响，防止不同算例/费用规则混用历史割与下界。
def fingerprint(m):
    """Bind recoverable cuts to the canonical model and its scenario identities."""
    a = m.A.copy(); a.sort_indices()
    digest = hashlib.sha256(b"suc-model-v1")
    for values, dtype in ((a.shape, "<i8"), (a.indptr, "<i8"), (a.indices, "<i8"),
                          (a.data, "<f8"), (m.rhs, "<f8"), (m.eq, "<f8"),
                          (m.obj, "<f8"), (m.q, "<f8"), (m.lb, "<f8"), (m.ub, "<f8"),
                          (m.binary, "<i8"), (m.time, "<i8")):
        values = np.asarray(values, dtype=dtype)
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    digest.update(json.dumps([m.d.n_units, m.d.horizon, [(s.name, s.probability) for s in m.scenarios]],
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


# 按变量所属小时划分块，返回块模型列表及跨块约束行号；跨块行移入 lambda*(A*x-rhs)。
# 价格作用于各块线性目标，右端常数只分配给首个涉及块，确保汇总时没有重复或漏计。
def partition(m, hours, prices=None):
    prices = np.zeros(len(m.rhs)) if prices is None else np.asarray(prices).copy()
    if (prices[~m.eq] < -1e-10).any():
        raise ValueError("Inequality multipliers must be nonnegative")
    prices[~m.eq] = np.maximum(0, prices[~m.eq])
    # 根据每条稀疏行涉及变量的首/末所属块，识别只能在协调层处理的跨块约束。
    owner = m.time // hours
    row, col = m.A.nonzero()
    first, last = np.full(len(m.rhs), owner.max() + 1), np.full(len(m.rhs), -1)
    np.minimum.at(first, row, owner[col]); np.maximum.at(last, row, owner[col])
    cross = np.flatnonzero(first != last)
    # 拉格朗日项 lambda @ (A_cross*x-rhs_cross) 的变量部分并入各块线性费用。
    costs = m.obj + m.A[cross].T @ prices[cross]
    blocks = []
    for j in range(owner.max() + 1):
        cols = np.flatnonzero(owner == j)
        rows = np.flatnonzero((first == j) & (last == j))
        times = np.unique(m.time[cols])
        b = NS(A=m.A[rows][:, cols], rhs=m.rhs[rows], eq=m.eq[rows], obj=costs[cols],
               q=m.q[cols], lb=m.lb[cols], ub=m.ub[cols], cols=cols, rows=rows,
               time=m.time[cols], times=times, d=m.d, scenarios=m.scenarios,
               constant=-float(m.rhs[cross[first[cross] == j]] @ prices[cross[first[cross] == j]]),
               crossA=m.A[cross][:, cols], crossrhs=m.rhs[cross] * (first[cross] == j),
               original_obj=m.obj[cols], labels=[m.labels[k] for k in rows], cross=cross)
        b.binary = np.flatnonzero(np.isin(cols, m.binary))
        b.continuous = np.flatnonzero(~np.isin(cols, m.binary))
        b.scale = max(1., float(np.sum(np.abs(m.obj[cols]) * b.ub + b.q * b.ub ** 2)))
        mapping = np.full(len(m.obj), -1); mapping[cols] = np.arange(len(cols))
        for name in ("u", "v", "w"):
            setattr(b, name, mapping[getattr(m.ix, name)[:, times]])
        blocks.append(b)
    return blocks, cross
