"""修复上限实验：记录纯 SA 的原始全天候选，离线比较现有修复与精确最小改动修复（P0）。

Gurobi 只作诊断标尺，用来回答"若修复理想，SA 样本能否成为改进"；对照方法仍只有无 LLM 的 SA。
"""
# 模块职责：原始候选的记录包装器、样本选取、P0 两阶段求解、块内上限与 Hamming 球对照、分类与判据。
from contextlib import contextmanager
import gzip
import json
from time import perf_counter
from types import SimpleNamespace as NS

import gurobipy as gp
import numpy as np
from scipy import sparse

from suc_benders import search as search_core
from suc_benders.benders import dispatch
from suc_benders.data import jsonable
from suc_benders.validation import commitment, commitment_hash, pack, repair, unpack, validate

# 与 optimization.optimize 相同的单线程参数；整数容差收紧到 1e-9，取整后再独立调度校验。
SETTINGS = dict(Threads=1, MIPGap=1e-6, FeasibilityTol=1e-8, OptimalityTol=1e-8, BarConvTol=1e-10, Seed=7,
                IntFeasTol=1e-9)
STATUS = {2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT',
          11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL'}
SELECT_STREAM = 20261028
REL_TOLERANCE = 1e-6


# 0/1 矩阵按行主序压成 hex 串；解码时按形状截去补齐位。
def u_to_bits(u):
    return np.packbits(np.asarray(u, dtype=np.uint8).ravel(order='C')).tobytes().hex()


def bits_to_u(text, shape):
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))
    return bits[:shape[0]*shape[1]].reshape(shape).astype(float)


# 在类级别包装 WholeDay.prepare：原函数之前补记原始 u、当时的现任方案与 SA 批次锚点，不改变任何求解行为。
# 初始化阶段尚无现任方案，不记录；现任方案的位串按身份只存一次。退出时恢复原方法。
@contextmanager
def log_proposals():
    original = search_core.WholeDay.prepare
    holder = NS(incumbents={})

    def prepare(self, raw, source, record):
        result = self.result
        if result.u is not None:
            values = np.asarray(raw, dtype=float)
            if values.shape == result.u.shape and np.isin(values, (0, 1)).all():
                record['raw_bits'] = u_to_bits(values)
            identity = commitment_hash(result)
            if identity not in holder.incumbents:
                holder.incumbents[identity] = dict(bits=u_to_bits(result.u), objective=float(result.objective))
            record.update(incumbent_id=identity, incumbent_objective=float(result.objective))
            if source == 'sa' and self.diagnostics['sampling']:
                record['anchor_cost'] = float(self.diagnostics['sampling'][-1]['anchor_cost'])
        return original(self, raw, source, record)

    search_core.WholeDay.prepare = prepare
    try:
        yield holder
    finally:
        search_core.WholeDay.prepare = original


# 带原始位串的提议逐条连接首次全天评估（按 commitment_id），供离线重放与描述统计。
def collect(whole):
    first = {}
    for e in whole.diagnostics['evaluations']:
        first.setdefault(e['commitment_hash'], e)
    rows = []
    for order, r in enumerate(whole.diagnostics['proposals']):
        if 'raw_bits' not in r:
            continue
        e = first.get(r.get('commitment_id'))
        rows.append(dict(order=order, **{k: r.get(k) for k in (
            'proposal_id', 'origin', 'block', 'round', 'outcome', 'raw_sha256', 'raw_bits', 'raw_discrete_feasible',
            'raw_fixed_bit_violations', 'hamming_u', 'commitment_id', 'incumbent_id', 'incumbent_objective',
            'anchor_cost')},
            evaluation=None if e is None else dict(feasible=e['feasible'], objective=e['objective'],
                                                   dispatch_status=e['dispatch_status'])))
    return rows


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False)+'\n')


def read_jsonl_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


# 选取重放样本：按原始哈希去重并保留重复次数，剔除与当时现任方案相同的样本，按块按比例（最大余数）分层，
# 块内随机抽取，最后整体随机排序，使中途停止时已重放部分仍近似均匀。返回 (样本, 计数)。
def select(rows, incumbents, shape, n, seed, origin='sa'):
    unique = {}
    for r in rows:
        if r['origin'] != origin:
            continue
        if r['raw_sha256'] in unique:
            unique[r['raw_sha256']]['multiplicity'] += 1
        else:
            unique[r['raw_sha256']] = dict(r, multiplicity=1)
    candidates, identical = [], 0
    for r in unique.values():
        anchor = bits_to_u(incumbents[r['incumbent_id']]['bits'], shape)
        if np.array_equal(bits_to_u(r['raw_bits'], shape), anchor):
            identical += 1
        else:
            candidates.append(r)
    rng = np.random.default_rng([seed, SELECT_STREAM])
    blocks = sorted({r['block'] for r in candidates}, key=lambda b: (b is None, b))
    groups = {b: sorted((r for r in candidates if r['block'] == b), key=lambda r: r['order']) for b in blocks}
    take = min(n, len(candidates))
    exact = {b: take*len(groups[b])/len(candidates) for b in blocks} if candidates else {}
    quota = {b: int(np.floor(exact[b])) for b in blocks}
    for b in sorted(blocks, key=lambda b: (-(exact[b]-quota[b]), str(b)))[:take-sum(quota.values())]:
        quota[b] += 1
    chosen = []
    for b in blocks:
        chosen += [groups[b][i] for i in rng.permutation(len(groups[b]))[:quota[b]]]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    return chosen, dict(proposals=sum(r['origin'] == origin for r in rows), unique=len(unique),
                        identical_to_incumbent=identical, eligible=len(candidates), selected=len(chosen),
                        per_block={str(b): quota[b] for b in blocks})


# Hamming(u, raw) = coef @ x + offset，只有 u 列非零；u 列按列主序（机组变化最快），不能直接展开 raw。
def hamming_coefficients(m, raw):
    raw = np.asarray(raw, dtype=float)
    coef = np.zeros(len(m.obj))
    coef[m.ix.u] = 1-2*raw
    return coef, float(raw.sum())


# 固定 u 后全天调度、恢复完整解并独立校验；返回已校验费用（unpack 口径），失败时保留原因。
def evaluate_u(m, full, u, seconds=60.):
    started = perf_counter()
    try:
        r = dispatch(full, pack(m.d, np.asarray(u, dtype=float)), seconds, certified=True)
        if not r.feasible:
            return dict(feasible=False, status=r.status, objective=None, seconds=perf_counter()-started)
        plan = unpack(m, r.point)
        validate(m.d, m.scenarios, plan)
        return dict(feasible=True, status=r.status, objective=plan.objective, dispatch_cost=float(r.cost),
                    commitment_id=commitment_hash(plan), seconds=perf_counter()-started)
    except (RuntimeError, ValueError) as error:
        return dict(feasible=None, status='ERROR', objective=None, message=f'{type(error).__name__}: {error}',
                    seconds=perf_counter()-started)


# 复算现有 repair()（确定性）并调度校验；记录其改动位数与方案身份，供与运行时记录核对。
def current_repair(m, full, raw):
    raw = np.asarray(raw, dtype=float)
    started = perf_counter()
    x = repair(m.d, m.scenarios, raw)
    seconds = perf_counter()-started
    if x is None:
        return dict(outcome='repair_failed', repair_seconds=seconds, feasible=False, objective=None)
    z = commitment(m.d, x)
    result = evaluate_u(m, full, z.u)
    return dict(outcome='repaired', repair_seconds=seconds, commitment_id=commitment_hash(z),
                hamming=int(np.count_nonzero(z.u != raw)), u_bits=u_to_bits(z.u), **{
                    k: v for k, v in result.items() if k != 'commitment_id'})


# 在完整模型上建立单线程 Gurobi 模型（binary 列为整数），可选覆盖变量上下界。
@contextmanager
def _model(m, lb=None, ub=None):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as g:
            for name, value in SETTINGS.items():
                g.setParam(name, value)
            types = np.full(len(m.obj), gp.GRB.CONTINUOUS)
            types[m.binary] = gp.GRB.BINARY
            x = g.addMVar(len(m.obj), lb=m.lb if lb is None else lb, ub=m.ub if ub is None else ub, vtype=types)
            g.addMConstr(m.A, x, np.where(m.eq, '=', '<'), m.rhs)
            yield g, x


def _cost(m, x):
    return m.obj @ x + x @ sparse.diags(m.q) @ x


def _solve(g, x, seconds):
    g.setParam('TimeLimit', max(float(seconds), 1e-2))
    started = perf_counter()
    g.optimize()
    out = dict(status=STATUS.get(g.Status, f'STATUS_{g.Status}'), seconds=perf_counter()-started,
               objective=float(g.ObjVal) if g.SolCount else None, bound=None)
    try:
        out['bound'] = float(g.ObjBound)
    except (AttributeError, gp.GurobiError):
        pass
    return out, (np.asarray(x.X) if g.SolCount else None)


# P0：按字典序的精确最小改动修复。阶段 1 求最小 Hamming 距离 h*；阶段 2 在 Hamming <= h*+slack 内
# 求真实成本最低（以阶段 1 解作起点），取整后独立调度校验。阶段 1 未证最优时不再继续。
def minimal_change(m, full, raw, *, slack=0, phase1=30., phase2=60.):
    raw = np.asarray(raw, dtype=float)
    coef, offset = hamming_coefficients(m, raw)
    out = dict(slack=slack)
    with _model(m) as (g, x):
        g.setObjective(coef @ x + offset)
        first, point = _solve(g, x, phase1)
        out.update(phase1_status=first['status'], phase1_seconds=first['seconds'])
        if first['status'] != 'OPTIMAL' or point is None:
            out['h_upper'] = None if point is None else int(round(first['objective']))
            return out
        h = int(round(first['objective']))
        if int(np.count_nonzero(np.rint(point[m.ix.u]) != raw)) != h:
            raise RuntimeError('Hamming objective and rounded projection disagree')
        out['h_star'] = h
        x.Start = point
        g.addConstr(coef @ x <= h+slack-offset)
        g.setObjective(_cost(m, x))
        second, point = _solve(g, x, phase2)
        out.update(phase2_status=second['status'], phase2_seconds=second['seconds'],
                   mip_objective=second['objective'], bound=second['bound'])
    if point is None:
        return out
    u = np.rint(point[m.ix.u])
    out.update(hamming=int(np.count_nonzero(u != raw)), u_bits=u_to_bits(u), **evaluate_u(m, full, u))
    return out


# 对照：块外 u 固定为现任方案（hours 为块内小时），或在现任方案的 Hamming 球内，求真实成本最低的方案。
def block_ceiling(m, full, anchor_u, hours, seconds=120.):
    anchor_u = np.asarray(anchor_u, dtype=float)
    lb, ub = m.lb.copy(), m.ub.copy()
    outside = np.setdiff1d(np.arange(m.d.horizon), hours)
    lb[m.ix.u[:, outside]] = ub[m.ix.u[:, outside]] = anchor_u[:, outside]
    with _model(m, lb, ub) as (g, x):
        x.Start = np.r_[pack(m.d, anchor_u), np.full(len(m.obj)-3*m.d.n_units*m.d.horizon, gp.GRB.UNDEFINED)]
        g.setObjective(_cost(m, x))
        out, point = _solve(g, x, seconds)
    if point is not None:
        out.update(evaluate_u(m, full, np.rint(point[m.ix.u])), mip_objective=out['objective'])
    return out


def ball(m, full, anchor_u, radius, seconds=120.):
    coef, offset = hamming_coefficients(m, anchor_u)
    with _model(m) as (g, x):
        x.Start = np.r_[pack(m.d, np.asarray(anchor_u, dtype=float)),
                        np.full(len(m.obj)-3*m.d.n_units*m.d.horizon, gp.GRB.UNDEFINED)]
        g.addConstr(coef @ x <= radius-offset)
        g.setObjective(_cost(m, x))
        out, point = _solve(g, x, seconds)
    if point is not None:
        out.update(evaluate_u(m, full, np.rint(point[m.ix.u])), mip_objective=out['objective'])
    return out


# 相对现任方案分类：阶段 1 未证最优一律 unresolved；已校验费用严格低于阈值为 improving；
# 阶段 2 最优，或其下界不低于阈值，为 non_improving；其余 unresolved。
def classify(p0, incumbent, rel=REL_TOLERANCE):
    threshold = incumbent-rel*abs(incumbent)
    if p0.get('phase1_status') != 'OPTIMAL':
        return 'unresolved'
    value = p0.get('objective')
    if value is not None and value < threshold:
        return 'improving'
    if value is not None and p0.get('phase2_status') == 'OPTIMAL':
        return 'non_improving'
    if p0.get('bound') is not None and p0['bound'] >= threshold:
        return 'non_improving'
    return 'unresolved'


# 预登记判据：classes 为 {种子: [每个已重放样本的分类]}，分母为已重放、与现任方案不同的去重样本。
def gate(classes, criteria):
    replayed = {s: len(c) for s, c in classes.items()}
    improving = {s: sum(v == 'improving' for v in c) for s, c in classes.items()}
    total = sum(replayed.values())
    unresolved = sum(v == 'unresolved' for c in classes.values() for v in c)
    rate = sum(improving.values())/total if total else 0.
    share = unresolved/total if total else 1.
    seeds_with = sum(count > 0 for count in improving.values())
    complete = len(classes) == criteria['seeds'] and min(replayed.values(), default=0) >= criteria['min_replayed']
    if seeds_with >= criteria['pass_seeds'] and rate >= criteria['pass_rate']:
        decision = 'PASS'
    elif seeds_with <= criteria['fail_seeds'] and share <= criteria['max_unresolved'] and complete:
        decision = 'FAIL'
    else:
        decision = 'INCONCLUSIVE'
    return dict(decision=decision, seeds_with_improving=seeds_with, improving_rate=rate, unresolved_share=share,
                replayed=replayed, improving=improving, total=total)


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


# 单次运行的描述统计（不进判据）：结局、距离、修复代价、改进幅度、加权改进率与时间分段。
def describe(oracle, incumbent, reference=None, tolerance=1e-4):
    gap = (lambda c: None if reference is None or c is None else 100*(c-reference)/abs(reference))
    sa = [r for r in oracle if r['kind'] == 'sa']
    both = [r for r in sa if r['current'].get('feasible') and r['p0'].get('feasible')]
    improving = [r for r in sa if r['class'] == 'improving']
    weight = sum(r['multiplicity'] for r in sa)
    rounds = sorted(r['round'] for r in sa)
    cuts = np.quantile(rounds, [1/3, 2/3]) if rounds else (0, 0)
    period = lambda r: 'early' if r['round'] <= cuts[0] else ('middle' if r['round'] <= cuts[1] else 'late')
    best = min((r['p0']['objective'] for r in sa if r['p0'].get('objective') is not None), default=None)
    return dict(
        replayed=len(sa), classes={c: sum(r['class'] == c for r in sa) for c in ('improving', 'non_improving', 'unresolved')},
        logged_outcomes={o: sum(r['logged_outcome'] == o for r in sa) for o in sorted({r['logged_outcome'] for r in sa})},
        raw_discrete_feasible=sum(bool(r['raw_discrete_feasible']) for r in sa),
        current_feasible=sum(bool(r['current'].get('feasible')) for r in sa),
        p0_feasible=sum(bool(r['p0'].get('feasible')) for r in sa),
        rescued=sum(not r['current'].get('feasible') and bool(r['p0'].get('feasible')) for r in sa),
        median_hamming_to_incumbent=_median([r['hamming_incumbent'] for r in sa]),
        median_h_current=_median([r['current'].get('hamming') for r in sa]),
        median_h_star=_median([r['p0'].get('h_star') for r in sa]),
        median_repair_tax=_median([r['current']['objective']-r['p0']['objective'] for r in both]),
        median_relative_to_incumbent_p0=_median([(r['p0']['objective']-incumbent)/abs(incumbent) for r in sa
                                                 if r['p0'].get('objective') is not None]),
        median_relative_to_incumbent_current=_median([(r['current']['objective']-incumbent)/abs(incumbent) for r in sa
                                                      if r['current'].get('objective') is not None]),
        best_p0_objective=best, best_p0_gain_percent=None if best is None else 100*(incumbent-best)/abs(incumbent),
        incumbent_gap_percent=gap(incumbent), best_p0_gap_percent=gap(best),
        improving_above_tolerance=sum((incumbent-r['p0']['objective']) > tolerance*abs(incumbent) for r in improving),
        weighted_improving_rate=sum(r['multiplicity'] for r in improving)/weight if weight else 0.,
        outside_block_changes=dict(median=_median([r['p0_outside_block_changes'] for r in sa]),
                                   positive=sum((r['p0_outside_block_changes'] or 0) > 0 for r in sa)),
        by_period={p: dict(replayed=sum(period(r) == p for r in sa),
                           improving=sum(period(r) == p and r['class'] == 'improving' for r in sa))
                   for p in ('early', 'middle', 'late')},
        current_improving=sum(r['current'].get('objective') is not None
                              and r['current']['objective'] < incumbent-REL_TOLERANCE*abs(incumbent) for r in sa),
        p0_seconds=_median([r['p0'].get('phase1_seconds', 0)+r['p0'].get('phase2_seconds', 0) for r in sa]))
