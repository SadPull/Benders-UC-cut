"""Time strengthening, explicit dispatch and perspective costs for integer proofs.

The input root must contain only valid master constraints and verified floors.
Predicted model objectives are not feasible upper bounds for the original SUC.
"""
from copy import copy
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .benders import master
from .model import fingerprint, partition
from .optimization import optimize


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
    proof = proof_model(physical)
    n = len(proof.obj)
    columns = np.r_[physical.binary, np.arange(n, n+len(root.obj)-nx)]
    count = n+len(root.obj)-nx
    constant = float(root.constant)
    out = NS(obj=np.zeros(count), q=np.zeros(count), lb=np.r_[proof.lb, root.lb[nx:]],
        ub=np.r_[proof.ub, root.ub[nx:]], binary=np.unique(np.r_[physical.binary, columns[root.binary]]),
        constant=constant, A=sparse.hstack([proof.A, sparse.csr_matrix((len(proof.rhs), count-n))], format="csr"),
        rhs=proof.rhs.copy(), eq=proof.eq.copy(), theta=int(columns[theta]), eta=proof.eta,
        physical_fingerprint=fingerprint(physical), purpose="strengthened_master")
    out.obj[out.theta] = 1.
    a = root.A.tocoo()
    mapped = sparse.csr_matrix((a.data, (a.row, columns[a.col])), shape=(len(root.rhs), count))
    out = _append(out, mapped, root.rhs, root.eq)
    cost = np.r_[proof.obj, np.zeros(count-n)]; cost[out.theta] -= 1.
    return _append(out, sparse.csr_matrix(cost[None, :]), [constant-proof.constant])


def separate(physical, strengthened, seconds, epsilon, *, callback=None):
    """Refine continuous costs until closed or timed out; never produces a UB."""
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite separation budget and tolerance required")
    started = perf_counter(); deadline = started+seconds
    result = NS(status="TIME_LIMIT", model=strengthened, lower=-np.inf, gap=np.inf, cuts=[])
    current = strengthened
    current_cuts = []
    while deadline-perf_counter() > .1:
        raw = optimize(current, deadline-perf_counter()-.1, callback=callback)
        if raw.status != "OPTIMAL" or not np.isfinite(raw.lower) or perf_counter() > deadline:
            break
        result.model, result.lower = current, float(raw.lower+current.constant)
        result.cuts = current_cuts.copy()
        gap, cuts = perspective_batch(physical, current, raw.x, epsilon)
        result.gap = gap
        if perf_counter() > deadline:
            break
        if not cuts:
            result.status = "SEPARATION_CLOSED"
            break
        current = add_perspectives(current, physical, cuts)
        current_cuts.extend(cuts)
    result.elapsed_seconds = perf_counter()-started
    return result


def certify(strengthened, seconds, *, callback=None):
    """One fixed-model MIPFocus=3 solve; expose ObjBound, not a feasible UB."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive finite certification budget required")
    started = perf_counter()
    raw = optimize(strengthened, seconds-min(3., seconds*.1), integral=True, mip_focus=3, callback=callback)
    elapsed = perf_counter()-started
    usable = (elapsed <= seconds and raw.status in ("OPTIMAL", "TIME_LIMIT", "INTERRUPTED", "USER_OBJ_LIMIT")
              and np.isfinite(raw.lower) and abs(raw.lower) < 1e100)
    lower = raw.lower+strengthened.constant-1e-5 if usable else -np.inf
    return NS(status=raw.status if elapsed <= seconds else "TIME_LIMIT", lower=float(lower),
              elapsed_seconds=elapsed)


def perspective_batch(physical, strengthened, point, epsilon):
    """Return the total perspective deficit and supports that separate the point."""
    # 整模型指纹每批只算一次，再传给每条割；逐割重算在默认算例上每轮可达数秒。
    identity = fingerprint(physical)
    if strengthened.physical_fingerprint != identity:
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
                    cuts.append(perspective_cut(physical, i, t, k, support, identity=identity))
    return gap, cuts if gap > epsilon else []


def add_perspectives(strengthened, physical, cuts):
    """Append only reconstructed, model-bound perspective inequalities."""
    identity = fingerprint(physical)
    if strengthened.physical_fingerprint != identity:
        raise ValueError("Perspective model identity changed")
    I, J, V = [], [], []
    for j, cut in enumerate(cuts):
        if cut.get("kind") != "perspective":
            raise ValueError("Expected perspective cuts")
        verify_cut(physical, cut, identity)
        I.extend([j]*3); J.extend([cut["p"], cut["u"], strengthened.eta[cut["p"]]])
        V.extend([cut["p_coefficient"], cut["u_coefficient"], -1.])
    A = sparse.csr_matrix((V, (I, J)), shape=(len(cuts), len(strengthened.obj)))
    return _append(strengthened, A, np.zeros(len(cuts)))


# 保留原物理约束，抬升块容量/AND 结构，并可用有界 eta 和有限透视支撑替代正二次项。
# 结果是整数有效的连续下估计模型，不保证其松弛值一定优于原连续 QP。
def proof_model(m):
    """Retain every physical row and lift the actual block capacity/AND rows.

    Eta replaces each positive quadratic term, with finite bounds and common
    pmin/midpoint/pmax supports. This is an integer-valid underestimator; its
    value is not assumed to dominate the original continuous QP.
    """
    if (np.any(m.q < 0)
            or not np.isfinite(m.lb).all() or not np.isfinite(m.ub).all()):
        raise ValueError('Proof model requires the bounded convex SUC model')
    out = copy(m)
    obj, q, lb, ub, time = [list(getattr(m, name)) for name in ('obj', 'q', 'lb', 'ub', 'time')]
    rows, rhs, labels = [], [], []
    for block in partition(m, m.d.horizon)[0]:
        native, _ = master(block, [])
        mapping = list(block.cols[block.binary])
        for z, v, w in native.and_pairs:
            assert z == len(mapping)
            mapping.append(len(obj))
            obj.append(0.); q.append(0.); lb.append(0.); ub.append(1.)
            time.append(int(m.time[mapping[v]]))
        for j, kind in enumerate(native.kind):
            if kind not in ('capacity', 'and'):
                continue
            row = native.A.getrow(j)
            rows.append({int(mapping[k]): float(v) for k, v in zip(row.indices, row.data)})
            rhs.append(float(native.rhs[j])); labels.append('proof_'+kind)
    eta = {}
    for k in range(len(m.scenarios)):
        for t in range(m.d.horizon):
            for i in range(m.d.n_units):
                p, u = int(m.ix.p[i, t, k]), int(m.ix.u[i, t])
                a = float(m.q[p])
                if a == 0:
                    continue
                e = len(obj); eta[p] = e
                q[p] = 0.
                obj.append(1.); q.append(0.); lb.append(0.); ub.append(a*m.ub[p]**2)
                time.append(t)
                for support in sorted(set((float(m.d.pmin[i]),
                        float((m.d.pmin[i]+m.d.pmax[i])/2), float(m.d.pmax[i])))):
                    rows.append({p: 2*a*support, u: -a*support**2, e: -1.})
                    rhs.append(0.); labels.append('perspective_initial')
    I, J, V = [], [], []
    for j, row in enumerate(rows):
        for k, value in row.items():
            if value:
                I.append(j); J.append(k); V.append(value)
    extra = sparse.coo_matrix((V, (I, J)), shape=(len(rows), len(obj))).tocsr()
    out.A = sparse.vstack([sparse.hstack([m.A, sparse.csr_matrix((len(m.rhs), len(obj)-len(m.obj)))]),
                           extra], format='csr')
    out.rhs, out.eq = np.r_[m.rhs, rhs], np.r_[m.eq, np.zeros(len(rows), bool)]
    for name, values in zip(('obj', 'q', 'lb', 'ub', 'time'), (obj, q, lb, ub, time)):
        setattr(out, name, np.asarray(values, dtype=int if name == 'time' else float))
    out.labels = list(m.labels)+labels
    out.eta = eta
    out.purpose = 'perspective_proof_lp'
    return out


# 从原模型和割的索引/支撑重新生成完整证明记录，逐字段检查模型身份与系数未被修改。
# identity 可传入调用方已算好的 fingerprint(m)，省去逐割重算。
def verify_cut(m, cut, identity=None):
    if cut.get('kind') != 'perspective':
        raise ValueError('Unknown strengthening cut')
    expected = perspective_cut(m, cut['unit'], cut['hour'], cut['scenario'], cut['support'], identity=identity)
    if cut != expected:
        raise ValueError('Strengthening cut model or coefficients changed')


# 在给定机组、小时、场景和支撑出力处生成 eta >= 2*a*support*p-a*support^2*u 的整数有效割。
# 支撑必须位于 Pmin/Pmax 之间，a 已包含场景概率；同时绑定原模型指纹。
def perspective_cut(m, unit, hour, scenario, support, *, identity=None):
    if (type(unit) is not int or not 0 <= unit < m.d.n_units
            or type(hour) is not int or not 0 <= hour < m.d.horizon
            or type(scenario) is not int or not 0 <= scenario < len(m.scenarios)):
        raise ValueError('Invalid perspective index')
    support = float(support)
    if not np.isfinite(support) or not m.d.pmin[unit] <= support <= m.d.pmax[unit]:
        raise ValueError('Invalid perspective support')
    p, u = int(m.ix.p[unit, hour, scenario]), int(m.ix.u[unit, hour])
    a = float(m.q[p])
    if a <= 0:
        raise ValueError('Perspective requires a positive weighted quadratic cost')
    return dict(kind='perspective', model_fingerprint=fingerprint(m) if identity is None else identity,
                unit=unit, hour=hour,
                scenario=scenario, support=support, p=p, u=u, a=a,
                p_coefficient=2*a*support, u_coefficient=-a*support**2,
                proof='a*(p-support)**2 >= 0 when u=1; p=0 when u=0')


def _append(m, A, rhs, eq=None):
    out = copy(m)
    out.A = sparse.vstack([m.A, A], format="csr")
    out.rhs = np.r_[m.rhs, rhs]
    out.eq = np.r_[m.eq, np.zeros(len(rhs), bool) if eq is None else eq]
    return out
