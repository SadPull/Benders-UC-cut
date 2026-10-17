"""Restricted whole-day SA masters; their bounds are search guidance only."""
from fractions import Fraction
from itertools import combinations, zip_longest

import numpy as np
from scipy import sparse

from .benders import master
from .validation import pack


def joint_master(full, cuts):
    """Keep the native master and add necessary full-horizon physical rows."""
    m, _ = master(full, cuts, search_reachability=True)
    d = full.d
    rows, cols, values, rhs, eq, kind = [], [], [], [], [], []
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


def select_units(movable, attempt, seed):
    """Seeded pair/triple coverage; improvements never reset the cursor."""
    if len(movable) < 3:
        return [int(i) for i in movable]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7103]))
    pairs, triples = [list(combinations(map(int, movable), size)) for size in (2, 3)]
    rng.shuffle(pairs); rng.shuffle(triples)
    order = [group for pair in zip_longest(pairs, triples) for group in pair if group is not None]
    return list(order[attempt % len(order)])
