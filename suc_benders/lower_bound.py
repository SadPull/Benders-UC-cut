"""Finite continuous proof models for the first lower-bound screening experiment."""
# 模块职责：生成可重建验证的透视割/容量覆盖割，并构造有限连续证明模型与候选割池。
# 割及下界来自数值证明；LLM 只能选择冻结候选编号，候选库结果不表示理论最优界。
from copy import copy
from fractions import Fraction

import numpy as np
from scipy import sparse

from .benders import master
from .model import fingerprint, partition


# 在给定机组、小时、场景和支撑出力处生成 eta >= 2*a*support*p-a*support^2*u 的整数有效割。
# 支撑必须位于 Pmin/Pmax 之间，a 已包含场景概率；同时绑定原模型指纹。
def perspective_cut(m, unit, hour, scenario, support):
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
    return dict(kind='perspective', model_fingerprint=fingerprint(m), unit=unit, hour=hour,
                scenario=scenario, support=support, p=p, u=u, a=a,
                p_coefficient=2*a*support, u_coefficient=-a*support**2,
                proof='a*(p-support)**2 >= 0 when u=1; p=0 when u=0')


# 若所选机组集的总 Pmax 严格不足以覆盖最坏场景负荷及备用，则要求补集至少一台机组开机。
# 容量比较使用 Fraction 精确算术，避免临界容量因浮点误差错误判定。
def cover_cut(m, hour, below_capacity_set):
    units = list(below_capacity_set)
    if (type(hour) is not int or not 0 <= hour < m.d.horizon
            or any(type(i) is not int or not 0 <= i < m.d.n_units for i in units)
            or len(set(units)) != len(units)):
        raise ValueError('Invalid cover indices')
    units.sort()
    capacity = sum((Fraction(str(m.d.pmax[i])) for i in units), Fraction(0))
    demand = (1+Fraction(str(m.d.reserve_fraction))) * max(
        Fraction(str(s.load[hour])) for s in m.scenarios)
    if capacity >= demand:
        raise ValueError('Cover does not have strictly insufficient capacity')
    complement = sorted(set(range(m.d.n_units))-set(units))
    return dict(kind='cover', model_fingerprint=fingerprint(m), hour=hour,
                below_capacity_set=units, complement=complement,
                capacity=str(capacity), demand=str(demand),
                indices=[int(m.ix.u[i, hour]) for i in complement],
                proof='binary feasible schedules need at least one unit outside the set')


# 从原模型和割的索引/支撑重新生成完整证明记录，逐字段检查模型身份与系数未被修改。
def verify_cut(m, cut):
    if cut.get('kind') == 'perspective':
        expected = perspective_cut(m, cut['unit'], cut['hour'], cut['scenario'], cut['support'])
    elif cut.get('kind') == 'cover':
        expected = cover_cut(m, cut['hour'], cut['below_capacity_set'])
    else:
        raise ValueError('Unknown strengthening cut')
    if cut != expected:
        raise ValueError('Strengthening cut model or coefficients changed')


# 保留原物理约束，抬升块容量/AND 结构，并可用有界 eta 和有限透视支撑替代正二次项。
# 结果是整数有效的连续下估计模型，不保证其松弛值一定优于原连续 QP。
def proof_model(m, *, perspective=True, cuts=(), block_hours=3):
    """Retain every physical row and lift the actual block capacity/AND rows.

    Eta replaces each positive quadratic term, with finite bounds and common
    pmin/midpoint/pmax supports. This is an integer-valid underestimator; its
    value is not assumed to dominate the original continuous QP.
    """
    if (getattr(m.d, 'model_profile', '') or np.any(m.q < 0)
            or not np.isfinite(m.lb).all() or not np.isfinite(m.ub).all()):
        raise ValueError('First-round proof model requires the bounded convex SUC model')
    for cut in cuts:
        verify_cut(m, cut)
    if not perspective and any(c['kind'] == 'perspective' for c in cuts):
        raise ValueError('Perspective cuts require eta variables')
    out = copy(m)
    obj, q, lb, ub, time = [list(getattr(m, name)) for name in ('obj', 'q', 'lb', 'ub', 'time')]
    rows, rhs, labels, auxiliary = [], [], [], []
    for block in partition(m, block_hours)[0]:
        native, _ = master(block, [])
        mapping = list(block.cols[block.binary])
        for z, v, w in native.and_pairs:
            assert z == len(mapping)
            index = len(obj)
            mapping.append(index)
            obj.append(0.); q.append(0.); lb.append(0.); ub.append(1.)
            time.append(int(m.time[mapping[v]]))
            auxiliary.append(dict(index=index, v=int(mapping[v]), w=int(mapping[w])))
        for j, kind in enumerate(native.kind):
            if kind not in ('capacity', 'and'):
                continue
            row = native.A.getrow(j)
            rows.append({int(mapping[k]): float(v) for k, v in zip(row.indices, row.data)})
            rhs.append(float(native.rhs[j])); labels.append('proof_'+kind)
    eta = {}
    if perspective:
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
    seen = set()
    for cut in cuts:
        key = (cut['kind'], cut['hour'], tuple(cut['below_capacity_set'])) if cut['kind'] == 'cover' else (
            cut['kind'], cut['unit'], cut['hour'], cut['scenario'], cut['support'])
        if key in seen:
            continue
        seen.add(key)
        if cut['kind'] == 'cover':
            rows.append(dict.fromkeys(cut['indices'], -1.)); rhs.append(-1.)
        else:
            rows.append({cut['p']: cut['p_coefficient'], cut['u']: cut['u_coefficient'],
                         eta[cut['p']]: -1.}); rhs.append(0.)
        labels.append(cut['kind']+'_selected')
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
    out.eta, out.and_lift = eta, auxiliary
    out.base_fingerprint = fingerprint(m)
    out.purpose = 'perspective_proof_lp' if perspective else 'structured_proof_qp'
    return out


# 在当前证明点上按归一化违反量筛选至多十六条透视割和十六条覆盖割，输出带编号候选池。
# 生成和评分只依赖连续解及精确容量证明，不调用整数调度或导入参考最优值。
def cut_pool(m, proof, point, hours):
    """At most 16 supports and 16 covers, scored without integer dispatch."""
    point = np.asarray(point)
    if (proof.base_fingerprint != fingerprint(m) or point.shape != proof.obj.shape
            or not np.isfinite(point).all()):
        raise ValueError('Cut pool point does not match its proof model')
    perspectives, covers = [], []
    for t in hours:
        t = int(t)
        for i in range(m.d.n_units):
            u = point[m.ix.u[i, t]]
            if u <= 1e-8:
                continue
            for k in range(len(m.scenarios)):
                p = int(m.ix.p[i, t, k])
                if p not in proof.eta:
                    continue
                support = float(np.clip(point[p]/u, m.d.pmin[i], m.d.pmax[i]))
                if support in (m.d.pmin[i], (m.d.pmin[i]+m.d.pmax[i])/2, m.d.pmax[i]):
                    continue
                cut = perspective_cut(m, i, t, k, support)
                violation = cut['p_coefficient']*point[p]+cut['u_coefficient']*u-point[proof.eta[p]]
                norm = np.linalg.norm([cut['p_coefficient'], cut['u_coefficient'], 1.])
                if violation > 1e-8:
                    perspectives.append((violation/norm, 3, cut))
        demand = (1+Fraction(str(m.d.reserve_fraction))) * max(
            Fraction(str(s.load[t])) for s in m.scenarios)
        order = sorted(range(m.d.n_units), key=lambda i: (-point[m.ix.u[i, t]], i))
        selected, capacity = [], Fraction(0)
        for i in order:
            size = Fraction(str(m.d.pmax[i]))
            if capacity+size < demand:
                selected.append(i); capacity += size
        if demand > 0:
            cut = cover_cut(m, t, selected)
            violation = 1-sum(point[j] for j in cut['indices'])
            if violation > 1e-8:
                covers.append((violation/np.sqrt(len(cut['indices'])), len(cut['indices']), cut))
    result = []
    for group in (perspectives, covers):
        for score, nonzeros, cut in sorted(group, key=lambda v: (-v[0], v[1]))[:16]:
            result.append(dict(id=len(result), score=float(score), nonzeros=nonzeros, cut=cut))
    return result


# 验证外部选择是冻结候选池中一到两个互不重复的整数编号，并返回对应已证明合法的割。
def select_ids(value, pool):
    if (not isinstance(value, list) or not 1 <= len(value) <= 2
            or any(type(i) is not int for i in value) or len(set(value)) != len(value)
            or any(i not in {row['id'] for row in pool} for i in value)):
        raise ValueError('Expected one or two distinct candidate IDs from the frozen pool')
    return [next(row['cut'] for row in pool if row['id'] == i) for i in value]
