"""Time strengthening, explicit dispatch and perspective costs for integer proofs.

The input root must contain only valid master constraints and verified floors.
Predicted model objectives are not feasible upper bounds for the original SUC.
"""
from copy import copy
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .lower_bound import perspective_cut, proof_model, verify_cut
from .model import fingerprint
from .optimization import optimize


def _append(m, A, rhs, eq=None):
    out = copy(m)
    out.A = sparse.vstack([m.A, A], format="csr")
    out.rhs = np.r_[m.rhs, rhs]
    out.eq = np.r_[m.eq, np.zeros(len(rhs), bool) if eq is None else eq]
    return out


def add_temporal(root, physical):
    """Append cumulative up/down rows, preserving the original horizon rules."""
    d, ix = physical.d, physical.ix
    mapping = {int(col):j for j, col in enumerate(physical.binary)}
    I, J, V, rhs = [], [], [], []
    for i in range(d.n_units):
        for t in range(d.horizon):
            up = ix.v[i, max(0, t-int(d.min_up[i])+1):t+1]
            down = ix.w[i, max(0, t-int(d.min_down[i])+1):t+1]
            for window, sign, bound in ((up, -1., 0.), (down, 1., 1.)):
                row = len(rhs)
                I.extend([row]*(len(window)+1))
                J.extend([*(mapping[int(col)] for col in window), mapping[int(ix.u[i, t])]])
                V.extend([*np.ones(len(window)), sign]); rhs.append(bound)
    A = sparse.csr_matrix((V, (I, J)), shape=(len(rhs), len(root.obj)))
    return _append(root, A, rhs)


def build(physical, root, theta):
    """Share u/v/w with an already time-strengthened Benders root."""
    nx = len(physical.binary)
    if (not nx <= theta < len(root.obj) or theta in root.binary or np.any(root.q)
            or not np.array_equal(np.flatnonzero(root.obj), [theta]) or root.obj[theta] != 1.):
        raise ValueError("Expected a linear master minimizing its theta variable")
    proof = proof_model(physical, block_hours=physical.d.horizon)
    n = len(proof.obj)
    columns = np.r_[physical.binary, np.arange(n, n+len(root.obj)-nx)]
    count = n+len(root.obj)-nx
    constant = float(getattr(root, "constant", 0.))
    out = NS(obj=np.zeros(count), q=np.zeros(count), lb=np.r_[proof.lb, root.lb[nx:]],
        ub=np.r_[proof.ub, root.ub[nx:]], binary=np.unique(np.r_[physical.binary, columns[root.binary]]),
        constant=constant, A=sparse.hstack([proof.A, sparse.csr_matrix((len(proof.rhs), count-n))], format="csr"),
        rhs=proof.rhs.copy(), eq=proof.eq.copy(), theta=int(columns[theta]), master_columns=columns,
        eta=proof.eta, and_lift=proof.and_lift, physical_size=len(physical.obj),
        physical_fingerprint=fingerprint(physical), purpose="strengthened_master")
    out.obj[out.theta] = 1.
    a = root.A.tocoo()
    mapped = sparse.csr_matrix((a.data, (a.row, columns[a.col])), shape=(len(root.rhs), count))
    out = _append(out, mapped, root.rhs, root.eq)
    cost = np.r_[proof.obj, np.zeros(count-n)]; cost[out.theta] -= 1.
    return _append(out, sparse.csr_matrix(cost[None, :]), [constant-proof.constant])


def perspective_batch(physical, strengthened, point, epsilon):
    """Return the total perspective deficit and supports that separate the point."""
    if strengthened.physical_fingerprint != fingerprint(physical):
        raise ValueError("Perspective model identity changed")
    point = np.asarray(point, float)
    if point.shape != strengthened.obj.shape or not np.isfinite(point).all() or epsilon <= 0:
        raise ValueError("Invalid perspective point or tolerance")
    gap, cuts = 0., []
    for k in range(len(physical.scenarios)):
        for t in range(physical.d.horizon):
            for i in range(physical.d.n_units):
                p = int(physical.ix.p[i, t, k])
                if p not in strengthened.eta:
                    continue
                u = float(point[physical.ix.u[i, t]])
                # shortcut: ignore u <= 1e-8, revisit for stricter separation tolerances.
                if u <= 1e-8:
                    continue
                support = float(np.clip(point[p]/u, physical.d.pmin[i], physical.d.pmax[i]))
                missing = max(0., float(physical.q[p]*point[p]**2/u-point[strengthened.eta[p]]))
                gap += missing
                if missing > 1e-8:
                    cuts.append(perspective_cut(physical, i, t, k, support))
    return gap, cuts if gap > epsilon else []


def add_perspectives(strengthened, physical, cuts):
    """Append only reconstructed, model-bound perspective inequalities."""
    if strengthened.physical_fingerprint != fingerprint(physical):
        raise ValueError("Perspective model identity changed")
    I, J, V = [], [], []
    for j, cut in enumerate(cuts):
        if cut.get("kind") != "perspective":
            raise ValueError("Expected perspective cuts")
        verify_cut(physical, cut)
        I.extend([j]*3); J.extend([cut["p"], cut["u"], strengthened.eta[cut["p"]]])
        V.extend([cut["p_coefficient"], cut["u_coefficient"], -1.])
    A = sparse.csr_matrix((V, (I, J)), shape=(len(cuts), len(strengthened.obj)))
    return _append(strengthened, A, np.zeros(len(cuts)))


def separate(physical, strengthened, seconds, epsilon, *, callback=None):
    """Refine continuous costs until closed or timed out; never produces a UB."""
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite separation budget and tolerance required")
    started = perf_counter(); deadline = started+seconds
    result = NS(status="TIME_LIMIT", model=strengthened, lower=-np.inf, gap=np.inf,
                iterations=0, added_cuts=0, cuts=[])
    current = strengthened
    current_cuts = []
    while deadline-perf_counter() > .1:
        raw = optimize(current, deadline-perf_counter()-.1, **({"callback": callback} if callback else {}))
        if raw.status != "OPTIMAL" or not np.isfinite(raw.lower) or perf_counter() > deadline:
            break
        result.model, result.lower = current, float(raw.lower+current.constant)
        result.cuts = current_cuts.copy()
        result.iterations += 1
        gap, cuts = perspective_batch(physical, current, raw.x, epsilon)
        result.gap = gap
        if perf_counter() > deadline:
            break
        if not cuts:
            result.status = "SEPARATION_CLOSED"
            break
        current = add_perspectives(current, physical, cuts)
        current_cuts.extend(cuts)
        result.added_cuts += len(cuts)
    result.elapsed_seconds = perf_counter()-started
    return result


def certify(strengthened, seconds, *, warm_start=None, target_lower=None, callback=None,
            log_file=None, model_file=None):
    """One fixed-model MIPFocus=3 solve; expose ObjBound, not a feasible UB."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive finite certification budget required")
    started = perf_counter()
    extra = {key: value for key, value in dict(callback=callback, log_file=log_file,
                                               model_file=model_file).items() if value is not None}
    raw = optimize(strengthened, seconds-min(3., seconds*.1), integral=True,
                   warm_start=warm_start, mip_focus=3, target_lower=target_lower, **extra)
    elapsed = perf_counter()-started
    usable = (elapsed <= seconds and raw.status in ("OPTIMAL", "TIME_LIMIT", "INTERRUPTED", "USER_OBJ_LIMIT")
              and np.isfinite(raw.lower) and abs(raw.lower) < 1e100)
    lower = raw.lower+strengthened.constant-1e-5 if usable else -np.inf
    return NS(status=raw.status if elapsed <= seconds else "TIME_LIMIT", lower=float(lower),
              point=raw.x, predicted_objective=raw.objval+strengthened.constant,
              elapsed_seconds=elapsed, optimizer_calls=1)
