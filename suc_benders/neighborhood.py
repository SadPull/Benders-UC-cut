"""Restricted whole-day SA masters; their bounds are search guidance only."""
from fractions import Fraction

import numpy as np
from scipy import sparse

from .benders import master
from .validation import pack, transition_allowed


def joint_master(full, cuts):
    """Keep the native master and add necessary full-horizon physical rows."""
    m, _ = master(full, cuts)
    d = full.d
    start, stop = transition_allowed(d)
    forbidden = np.r_[full.v[~start].ravel(), full.w[~stop]]
    rows, cols, values, rhs, eq, kind = [], [], [], [], [], []
    for column in forbidden:
        rows.append(len(rhs)); cols.append(int(column)); values.append(1.)
        rhs.append(0.); eq.append(True); kind.append('hard')
    minimum = np.min([s.load for s in full.scenarios], axis=0)
    for t in range(d.horizon):
        row = len(rhs)
        exact = {int(j): Fraction(str(p)) for j, p in zip(full.u[:, t], d.pmin) if p}
        rows.extend([row]*len(exact)); cols.extend(exact); values.extend(map(float, exact.values()))
        m.capacity_rows[len(m.rhs)+row] = (exact, -Fraction(str(minimum[t])))
        rhs.append(float(minimum[t])); eq.append(False); kind.append('capacity')
    m.A = sparse.vstack([m.A, sparse.csr_matrix((values, (rows, cols)),
                         shape=(len(rhs), len(m.obj)))], format='csr')
    m.rhs = np.r_[m.rhs, rhs]; m.eq = np.r_[m.eq, eq]; m.kind.extend(kind)
    m.purpose = 'joint_neighborhood_lp_relaxation'
    return m


def fix_units(m, full, u, units):
    """Fix only commitment columns; all continuous recourse remains in full."""
    selected = np.zeros(full.d.n_units, bool); selected[units] = True
    outside = np.r_[full.u[~selected].ravel(), full.v[~selected].ravel(), full.w[~selected].ravel()]
    x = pack(full.d, u)
    rows = sparse.csr_matrix((np.ones(len(outside)), (np.arange(len(outside)), outside)),
                             shape=(len(outside), len(m.obj)))
    m.A = sparse.vstack([m.A, rows], format='csr')
    m.rhs = np.r_[m.rhs, x[outside]]
    m.eq = np.r_[m.eq, np.ones(len(outside), bool)]
    m.kind.extend(['hard']*len(outside))


def select_units(d, u, movable, gradient, attempt, size, seed):
    """One coverage slot, then alternating economic donor/replacement slots."""
    order = np.random.default_rng(np.random.SeedSequence([seed, 7103])).permutation(movable).tolist()
    selected = [order[attempt % len(order)]]
    saving = np.zeros(d.n_units)
    for row in gradient:
        saving[row['unit']] += max(0., -row['estimated_total_change'])
    donors = sorted((i for i in movable if u[i].any()), key=lambda i: (-saving[i], int(i)))
    replacements = sorted((i for i in movable if d.pmax[i] > 0 and (u[i] == 0).any()),
        key=lambda i: (d.cost_a[i]*d.pmax[i]+d.cost_b[i]+d.cost_c[i]/d.pmax[i], int(i)))
    for slot in range(1, size):
        priority = donors if (slot+attempt) % 2 else replacements
        choice = next((i for i in [*priority, *order] if i not in selected), None)
        if choice is None:
            break
        selected.append(int(choice))
    return selected
