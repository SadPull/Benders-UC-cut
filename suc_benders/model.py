"""Sparse SUC model; all rows are A @ point <= rhs, or equality rows.

Variable ordering matches MATLAB: variable type, scenario, hour, unit.
Internal indices and hours are zero based; schedule matrices are unit x hour.
"""
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse


def model(d, scenarios):
    n, T, S = d.n_units, d.horizon, len(scenarios)
    N = n * T
    u = np.arange(N).reshape((n, T), order="F")
    p = np.arange(3 * N, (3 + S) * N).reshape((n, T, S), order="F")
    ix = NS(u=u, v=u + N, w=u + 2 * N, p=p, r=p + N * S)
    nv = (3 + 2 * S) * N
    m = NS(obj=np.zeros(nv), q=np.zeros(nv), lb=np.zeros(nv), ub=np.ones(nv), ix=ix,
           binary=np.arange(3 * N), time=np.tile(np.repeat(np.arange(T), n), 3 + 2 * S),
           d=d, scenarios=scenarios, constant=0.)
    m.obj[ix.u], m.obj[ix.v], m.obj[ix.w] = d.cost_c[:, None], d.startup_cost[:, None], d.shutdown_cost[:, None]
    for k, s in enumerate(scenarios):
        m.obj[ix.p[:, :, k]] = s.probability * d.cost_b[:, None]
        m.q[ix.p[:, :, k]] = s.probability * d.cost_a[:, None]
        m.ub[ix.p[:, :, k]] = d.pmax[:, None]
        m.ub[ix.r[:, :, k]] = d.pmax[:, None]
    I, J, V, rhs, eq, labels = [], [], [], [], [], []

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


def partition(m, hours, prices=None):
    prices = np.zeros(len(m.rhs)) if prices is None else np.asarray(prices).copy()
    if (prices[~m.eq] < -1e-10).any():
        raise ValueError("Inequality multipliers must be nonnegative")
    prices[~m.eq] = np.maximum(0, prices[~m.eq])
    owner = m.time // hours
    row, col = m.A.nonzero()
    first, last = np.full(len(m.rhs), owner.max() + 1), np.full(len(m.rhs), -1)
    np.minimum.at(first, row, owner[col]); np.maximum.at(last, row, owner[col])
    cross = np.flatnonzero(first != last)
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
               original_obj=m.obj[cols])
        b.binary = np.flatnonzero(np.isin(cols, m.binary))
        b.continuous = np.flatnonzero(~np.isin(cols, m.binary))
        b.scale = max(1., float(np.sum(np.abs(m.obj[cols]) * b.ub + b.q * b.ub ** 2)))
        mapping = np.full(len(m.obj), -1); mapping[cols] = np.arange(len(cols))
        for name in ("u", "v", "w"):
            setattr(b, name, mapping[getattr(m.ix, name)[:, times]])
        blocks.append(b)
    return blocks, cross
