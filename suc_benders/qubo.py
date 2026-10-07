"""Cost register and bounded slack encoding with one-sided cut quantization."""
import math
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse


def bit_weights(maximum):
    maximum = int(np.floor(maximum + .5))
    if not 0 <= maximum <= 2 ** 53:
        raise ValueError("Integer encoding exceeds exact double range")
    if not maximum:
        return np.array([], dtype=float)
    n = maximum.bit_length()
    return np.asarray([2 ** j for j in range(n - 1)] + [maximum - (2 ** (n - 1) - 1)], float)


def encode(value, weights):
    value = min(float(np.sum(weights)), max(0., value))
    result = np.zeros(len(weights))
    for j in range(len(weights) - 1, -1, -1):
        if value >= weights[j]:
            result[j] = 1; value -= weights[j]
    return result


def qubo(m, o, seeds, upper):
    nb = len(m.binary)
    fixed = np.full(nb, np.nan)
    changed = True
    while changed:
        changed = False
        for k, label in enumerate(m.kind):
            if label != "hard":
                continue
            row = m.A[k, :nb].tocsr()
            js, values = row.indices, row.data
            known = ~np.isnan(fixed[js])
            constant = -m.rhs[k] + values[known] @ fixed[js[known]]
            js, values = js[~known], values[~known]
            for j in range(len(js)):
                rest = np.delete(values, j)
                lo, hi = constant + np.minimum(0, rest).sum(), constant + np.maximum(0, rest).sum()
                allowed = np.array([lo, lo + values[j]]) <= 1e-9
                if m.eq[k]:
                    allowed &= np.array([hi, hi + values[j]]) >= -1e-9
                if not allowed.any():
                    raise ValueError("Inconsistent hard constraints")
                if allowed.sum() == 1:
                    fixed[js[j]] = int(allowed[1]); changed = True
                    break
    free, known = np.flatnonzero(np.isnan(fixed)), np.flatnonzero(~np.isnan(fixed))
    lower = np.floor(m.lb[-1] / o.theta_step) * o.theta_step
    upper = np.ceil(min(upper, m.ub[-1]) / o.theta_step) * o.theta_step
    if not np.isfinite(upper) or upper < lower:
        raise ValueError("Invalid epigraph range")
    weights = o.theta_step * bit_weights((upper - lower) / o.theta_step)
    theta_ids = np.arange(len(free), len(free) + len(weights))
    nv = len(free) + len(weights)
    offset = lower / m.scale
    qi, qj, qv = list(theta_ids), list(theta_ids), list(weights / m.scale)
    penalty_rows, constants, slacks = [], [], []
    for k, label in enumerate(m.kind):
        a = np.r_[m.A[k, free].toarray().ravel(), float(m.A[k, nb]) * weights]
        c = (m.A[k, known] @ fixed[known]).item() - m.rhs[k] + m.A[k, nb] * lower
        if label == "hard" and not m.eq[k] and np.count_nonzero(a) <= 2:
            ids = np.flatnonzero(a)
            e00 = max(0, c) ** 2; offset += o.penalty * e00
            if len(ids):
                e10 = max(0, c + a[ids[0]]) ** 2
                qi.append(ids[0]); qj.append(ids[0]); qv.append(o.penalty * (e10 - e00))
                if len(ids) == 2:
                    e01, e11 = max(0, c + a[ids[1]]) ** 2, max(0, c + a[ids].sum()) ** 2
                    mixed = o.penalty * (e11 - e10 - e01 + e00) / 2
                    qi.extend([ids[1], ids[0], ids[1]]); qj.extend([ids[1], ids[1], ids[0]])
                    qv.extend([o.penalty * (e01 - e00), mixed, mixed])
            continue
        if label == "hard":
            a, c, scale = np.rint(a), np.rint(c), 1.
        elif label == "capacity":
            raw = a * 1e6
            if (np.abs(raw - np.rint(raw)) >= 1e-5).any() or abs(c * 1e6 - round(c * 1e6)) >= 1e-4:
                raise ValueError("Capacity data supports at most six decimal places")
            ai = np.rint(raw)
            divisor = math.gcd(*(int(abs(v)) for v in ai if v)) or 1
            a, c = ai / divisor, np.ceil(round(c * 1e6) / divisor)
            scale = divisor / 1e6 / o.reserve_scale
        else:
            step, norm = (o.theta_step, m.scale) if label == "optimality" else (1e-3, o.reserve_scale)
            a, c, scale = np.floor(a / step), np.floor(c / step), step / norm
        low, high = c + np.minimum(0, a).sum(), c + np.maximum(0, a).sum()
        if not m.eq[k] and high <= 0:
            continue
        sw = bit_weights(max(0, -low)) if not m.eq[k] else np.array([])
        ids = np.arange(nv, nv + len(sw)); nv += len(sw)
        if not m.eq[k]:
            slacks.append((ids, sw, a, c))
        indices, values = np.r_[np.arange(len(a)), ids], np.r_[a, sw] * scale
        nonzero = values != 0
        penalty_rows.append((indices[nonzero], values[nonzero]))
        constants.append(c * scale)
    I, J, V = [], [], []
    for i, (indices, values) in enumerate(penalty_rows):
        I.extend([i] * len(indices)); J.extend(indices); V.extend(values)
    R = sparse.coo_matrix((V, (I, J)), shape=(len(constants), nv)).tocsr()
    c = np.asarray(constants)
    Q = sparse.coo_matrix((qv, (qi, qj)), shape=(nv, nv)).tocsr()
    Q = Q + o.penalty * (R.T @ R + sparse.diags(2 * (R.T @ c)))
    Q.eliminate_zeros()
    offset += o.penalty * (c @ c)
    starts = np.zeros((nv, len(seeds)))
    for k, seed in enumerate(seeds):
        x = np.zeros(nb); x[:m.nx] = seed
        for z, v, w in m.and_pairs:
            x[z] = x[v] * x[w]
        x[known] = fixed[known]
        rows = m.A[:, -1].toarray().ravel() < 0
        theta = max(lower, np.max(m.A[rows, :nb] @ x - m.rhs[rows], initial=-np.inf))
        starts[:len(free), k] = x[free]
        starts[theta_ids, k] = encode(np.ceil((min(upper, theta) - lower) / o.theta_step), weights / o.theta_step)
        for ids, sw, a, constant in slacks:
            starts[ids, k] = encode(np.floor(-constant - a @ starts[:len(a), k] + .5), sw)
    return NS(Q=Q.tocsr(), offset=float(offset), fixed=fixed, free=free, nx=m.nx, starts=starts,
              theta_ids=theta_ids, theta_weights=weights, theta_lower=lower, energy_unit=o.theta_step / m.scale)
