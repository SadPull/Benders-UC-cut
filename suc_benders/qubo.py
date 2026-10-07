"""Sparse assembly of archived penalties and whole-cut quantization budgets."""
from fractions import Fraction
from itertools import product
import math
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse


def bit_weights(maximum):
    maximum = int(maximum)
    if not 0 <= maximum <= 2 ** 53:
        raise ValueError("Integer encoding exceeds exact double range")
    if not maximum:
        return np.array([], dtype=float)
    n = maximum.bit_length()
    return np.asarray([2 ** j for j in range(n - 1)] + [maximum - (2 ** (n - 1) - 1)], float)


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


def fixed_binary(m):
    fixed = np.full(len(m.binary), np.nan)
    rows = []
    for k, label in enumerate(m.kind):
        row = m.A[k, :len(fixed)].tocsr()
        if label == "hard" and row.nnz <= 3:
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

    # Archived Rosenberg AND penalty, not three squared inequalities.
    for z, v, w in m.and_pairs:
        for i, j, coefficient in ((v, w, 1), (v, z, -2), (w, z, -2), (z, z, 3)):
            add_product(i, j, coefficient)

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
            continue
        sw = bit_weights(max(0, -low)) if not m.eq[k] else np.array([])
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
    Q = Q + o.penalty * (R.T @ R + sparse.diags(2 * (R.T @ c)))
    Q.eliminate_zeros()
    offset += o.penalty * (c @ c)
    return NS(Q=Q.tocsr(), offset=float(offset), fixed=fixed, free=free, nx=m.nx,
              theta_ids=theta_ids, theta_weights=weights, theta_lower=lower, energy_unit=o.theta_step / m.scale)
