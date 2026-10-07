"""Continuous recourse cuts, exact bound certification and dual coordination."""
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .optimization import optimize


def dispatch(b, x, seconds):
    started = perf_counter()
    X, Y = b.binary, b.continuous
    x = np.asarray(x)
    if x.shape != (len(X),) or not np.isin(x, [0, 1]).all():
        raise ValueError("Dispatch requires an exact binary vector")
    result = NS(feasible=False, status="DISCRETE_INFEASIBLE", cost=np.inf,
                point=None, cut=None, recourse=np.inf)
    hard = b.A[:, Y].getnnz(axis=1) == 0
    residual = b.A[hard][:, X] @ x - b.rhs[hard]
    residual[b.eq[hard]] = np.abs(residual[b.eq[hard]])
    if (residual > 1e-7).any():
        return result
    rows = np.flatnonzero(~hard)
    B = b.A[rows][:, X]
    g = NS(A=b.A[rows][:, Y], rhs=b.rhs[rows] - B @ x, eq=b.eq[rows],
           obj=b.obj[Y], q=b.q[Y], lb=b.lb[Y], ub=b.ub[Y], binary=[])
    raw = optimize(g, seconds)
    result.status = raw.status
    if raw.status == "OPTIMAL":
        beta = np.asarray(B.T @ raw.prices)
        result.cut = NS(beta=beta, alpha=raw.lower - beta @ x, optimality=True)
        result.point = np.zeros(len(b.obj)); result.point[X] = x; result.point[Y] = raw.x
        result.recourse = raw.objval
        result.cost = raw.objval + b.obj[X] @ x + b.constant
        result.feasible = True
    elif raw.status in ("INFEASIBLE", "INF_OR_UNBD") and perf_counter() - started < seconds:
        E = np.flatnonzero(g.eq)
        C = sparse.vstack([g.A, -g.A[E]], format="csr")
        D = sparse.vstack([B, -B[E]], format="csr")
        h = np.r_[b.rhs[rows], -b.rhs[rows[E]]]
        nr, ny = len(h), len(Y)
        elastic = NS(A=sparse.hstack([C, -sparse.eye(nr)], format="csr"), rhs=h - D @ x,
                     eq=np.zeros(nr, bool), obj=np.r_[np.zeros(ny), np.ones(nr)],
                     q=np.zeros(ny + nr), lb=np.r_[g.lb, np.zeros(nr)],
                     ub=np.r_[g.ub, np.full(nr, np.inf)], binary=[])
        raw = optimize(elastic, seconds - (perf_counter() - started))
        result.status = raw.status
        if raw.status == "OPTIMAL":
            prices = np.clip(raw.prices, 0, 1)
            c = C.T @ prices
            z = np.where(c < 0, g.ub, g.lb)
            result.cut = NS(beta=np.asarray(D.T @ prices), alpha=float(-h @ prices + c @ z - 1e-7), optimality=False)
            result.recourse, result.status = raw.objval, "INFEASIBLE"
    return result


def master(b, cuts, seconds=0):
    X, Y = b.binary, b.continuous
    nx = len(X)
    hard = b.A[:, Y].getnnz(axis=1) == 0
    A = b.A[hard][:, X].tocsr()
    rows = [dict(zip(A.indices[A.indptr[j]:A.indptr[j + 1]], A.data[A.indptr[j]:A.indptr[j + 1]])) for j in range(A.shape[0])]
    rhs, eq = list(b.rhs[hard]), list(b.eq[hard])
    kind = ["hard"] * len(rows)
    mapping = np.full(len(b.obj), -1); mapping[X] = np.arange(nx)
    pairs, nb = [], nx

    def add(indices, values, bound, equality=False, label="hard"):
        rows.append({int(j): float(v) for j, v in zip(indices, values) if v != 0})
        rhs.append(bound); eq.append(equality); kind.append(label)

    for t, hour in enumerate(b.times):
        capacity = {}
        for i in range(b.d.n_units):
            u, v = mapping[b.u[i, t]], mapping[b.v[i, t]]
            sr, dr = b.d.pmax[i] - b.d.pstart[i], b.d.pmax[i] - b.d.pshut[i]
            capacity[u] = b.d.pmax[i]; capacity[v] = -sr
            if t < len(b.times) - 1:
                w = mapping[b.w[i, t + 1]]; capacity[w] = -dr
                if min(sr, dr) > 0:
                    z = nb; nb += 1; capacity[z] = min(sr, dr)
                    pairs.append((z, v, w))
                    add([z, v], [1, -1], 0); add([z, w], [1, -1], 0)
                    add([v, w, z], [1, 1, -1], 1)
        required = (1 + b.d.reserve_fraction) * max(s.load[hour] for s in b.scenarios)
        add(capacity.keys(), [-v for v in capacity.values()], -required, label="capacity")
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
           binary=np.arange(nb), nx=nx, and_pairs=pairs, kind=kind, scale=b.scale)
    lower = -np.inf
    if seconds > 0:
        raw = optimize(m, seconds, True)
        if raw.status == "INFEASIBLE":
            raise RuntimeError("Certified master is infeasible")
        lower = raw.lower + b.constant - 1e-5
    # The certification solution is deliberately not exposed to the search.
    return m, lower


def coordinate(planes, equality, incumbent, seconds):
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
    raw = optimize(m, seconds)
    if raw.status != "OPTIMAL":
        raise RuntimeError(f"Coordination LP failed: {raw.status}")
    prices = raw.x[:nc]
    prices[~equality] = np.maximum(0, prices[~equality])
    return prices, -raw.objval
