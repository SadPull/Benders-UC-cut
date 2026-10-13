"""Independent finite-bound proof checks retained from the experiment auditors."""
import numpy as np
from scipy import sparse


def verify_cut(b, cut):
    """Weak duality over finite recourse bounds; no solver or sampled commitments needed."""
    X, Y = b.binary, b.continuous
    rows = np.flatnonzero(b.A[:, Y].getnnz(axis=1))
    A, B, rhs, eq = b.A[rows][:, Y], b.A[rows][:, X], b.rhs[rows], b.eq[rows]
    proof = cut['proof']
    if cut['optimality']:
        if proof['kind'] != 'bounded_lagrangian':
            raise ValueError('Wrong optimality proof')
    else:
        if proof['kind'] != 'bounded_farkas':
            raise ValueError('Wrong feasibility proof')
        E = np.flatnonzero(eq)
        A, B = sparse.vstack([A, -A[E]], format='csr'), sparse.vstack([B, -B[E]], format='csr')
        rhs, eq = np.r_[rhs, -rhs[E]], np.zeros(len(rhs)+len(E), bool)
    prices = np.zeros(len(rhs))
    indices = np.asarray(proof['indices'], int)
    if (len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= len(rhs)).any()):
        raise ValueError('Invalid dual indices')
    prices[indices] = proof['prices']
    if not np.isfinite(prices).all() or (prices[~eq] < 0).any():
        raise ValueError('Invalid dual multiplier sign')
    c = np.asarray(A.T @ prices)+(b.obj[Y] if cut['optimality'] else 0)
    q = b.q[Y] if cut['optimality'] else np.zeros(len(Y))
    z = np.where(c < 0, b.ub[Y], b.lb[Y]).copy()
    curved = q > 0
    z[curved] = np.clip(-c[curved]/(2*q[curved]), b.lb[Y][curved], b.ub[Y][curved])
    alpha = float(c @ z+q @ z**2-rhs @ prices)
    beta = np.asarray(B.T @ prices)
    if (not np.isfinite(alpha) or not np.isfinite(cut['alpha'])
            or cut['alpha'] > alpha+1e-6 or not np.allclose(beta, cut['beta'], atol=1e-7, rtol=1e-12)):
        raise ValueError('Saved cut is not implied by its dual certificate')


def quadratic_dual(m, prices):
    prices = np.asarray(prices)
    if prices.shape != m.rhs.shape or not np.isfinite(prices).all() or (prices[~m.eq] < 0).any():
        raise ValueError('Invalid initial dual certificate')
    c = m.obj+np.asarray(m.A.T @ prices)
    z = np.where(c < 0, m.ub, m.lb).copy()
    curved = m.q > 0
    z[curved] = np.clip(-c[curved]/(2*m.q[curved]), m.lb[curved], m.ub[curved])
    return float(np.sum(c*z+m.q*z*z)-m.rhs @ prices)
