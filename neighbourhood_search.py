"""邻域实验：选出“机组 × 小时”邻域，SA+Benders 在全天块上只放开该邻域求解。

实验代码，不属于生产求解器（suc_benders/ 不改）。邻域外的 u 固定为当前方案；SA 采样受限 QUBO，
全天调度学习 Benders 割（割库跨邻域保留），候选照常经全天修复、调度和独立校验后才能更新 UB。
Gurobi 只作诊断尺子（邻域内精确最优、至多改 k 台的最优），参考解只用来计算差距。
"""
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace as NS

import gurobipy as gp
import numpy as np
from scipy import sparse

from offline_pilot import necessary
from suc_benders import search as search_core, solver
from suc_benders.anneal import anneal
from suc_benders.benders import discrete_feasible, dispatch, hard_rows
from suc_benders.model import model, partition
from suc_benders.validation import commitment_hash, pack, unpack, validate

# 每个邻域至多放开的机组数。
MAX_UNITS = 12
SOURCE = 'nsa'
# 与 repair_headroom（v1.3.0 归档）相同的单线程 Gurobi 参数；整数容差收紧，取整后再独立调度校验。
SETTINGS = dict(Threads=1, MIPGap=1e-6, FeasibilityTol=1e-8, OptimalityTol=1e-8, BarConvTol=1e-10, Seed=7,
                IntFeasTol=1e-9)
STATUS = {2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT',
          11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL'}
REL_TOLERANCE = 1e-6


# 邻域内部表示：units 为 0 起始机组号（升序、不重复），[start, end) 为 0 起始的半开小时区间。
def neighbourhood(units, start, end):
    return NS(units=tuple(sorted(int(i) for i in units)), start=int(start), end=int(end))


def describe(n):
    return dict(units=[i+1 for i in n.units], hours=[n.start+1, n.end])


# 程序输出格式：{"units": [1 起始机组号], "hours": [首小时, 末小时]（1 起始、含首尾）, "reason": 可选}。
# 不合格时抛出 ValueError，原因写在消息里；调用方只剔除这一个邻域。
def parse_neighbourhood(item, d, max_units=MAX_UNITS):
    if not isinstance(item, dict) or set(item) - {'units', 'hours', 'reason'} or not {'units', 'hours'} <= set(item):
        raise ValueError('neighbourhood must be a dict with units, hours and optional reason')
    units, hours = item['units'], item['hours']
    if not isinstance(units, (list, tuple)) or not 1 <= len(units) <= max_units:
        raise ValueError(f'units must list 1 to {max_units} units')
    if any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= d.n_units for i in units):
        raise ValueError('unit numbers must be integers from 1 to n_units')
    if len(set(units)) != len(units):
        raise ValueError('duplicate units')
    if (not isinstance(hours, (list, tuple)) or len(hours) != 2
            or any(isinstance(h, bool) or not isinstance(h, int) for h in hours)
            or not 1 <= hours[0] <= hours[1] <= d.horizon):
        raise ValueError('hours must be [first, last] with 1 <= first <= last <= horizon')
    return neighbourhood([i-1 for i in units], hours[0]-1, hours[1])


# 放开的位：邻域机组在 [start, end) 的 u；转移行 u_t - v_t + w_t - u_{t-1} = 0 使窗口后一小时的 v/w 也须放开。
def free_masks(d, n):
    u = np.zeros((d.n_units, d.horizon), bool)
    vw = np.zeros_like(u)
    rows = list(n.units)
    u[np.ix_(rows, range(n.start, n.end))] = True
    vw[np.ix_(rows, range(n.start, min(n.end+1, d.horizon)))] = True
    return u, vw


# 全天块上的持久 SA+Benders：整个运行只建一个，每次 focus 只放开一个邻域，块外（邻域外）位固定为当前方案。
# 局部调度就是全天调度，学到的割对全天成立，因此割库跨邻域保留。
class NeighbourhoodSearch(search_core.BlockSearch):
    def __init__(self, whole, o):
        super().__init__(whole.full, o, 0, whole)
        b = self.b
        self.base_fixed = self.fixed.copy()
        position = np.full(len(b.obj), -1); position[b.binary] = np.arange(len(b.binary))
        self.position = NS(u=position[b.u], v=position[b.v], w=position[b.w])
        self.current = None
        self.diagnostic.update(kind='neighbourhood')

    # 按当前 incumbent 固定邻域外的位并重建硬约束行；返回放开（未被固定）的二进制位数。
    def focus(self, n):
        whole = self.whole
        free_u, free_vw = free_masks(whole.d, n)
        incumbent = np.rint(whole.best.point[self.b.cols[self.b.binary]])
        fixed = self.base_fixed.copy()
        for name, free in (('u', free_u), ('v', free_vw), ('w', free_vw)):
            index = getattr(self.position, name)[~free]
            fixed[index] = np.where(np.isfinite(fixed[index]), fixed[index], incumbent[index])
        self.fixed, self.hard = fixed, hard_rows(self.b, fixed=fixed)
        self.current, self.free_u = n, free_u
        return int(np.isnan(fixed).sum())

    # 在当前邻域内采样一批 SA：先评估 incumbent（种子），再评估每个原始样本学割，
    # 拼成全天方案后走全天修复、调度和校验。返回本批的标量记录。
    def step(self, k, label=SOURCE):
        b, o, whole = self.b, self.o, self.whole
        if whole.remaining() <= 0 or self.current is None:
            return None
        columns = b.cols[b.binary]
        anchor = np.rint(whole.best.point[columns])
        self.evaluate(anchor, 'initial')
        if whole.remaining() <= 0:
            return None
        baseline_u, before = whole.result.u.copy(), whole.result.objective
        anchor_id = commitment_hash(whole.result)
        phase = perf_counter()
        q, stored, encoded = self.build_qubo(anchor, [])
        whole.timings['qubo'] += perf_counter()-phase
        if whole.remaining() <= 0:
            return None
        phase = perf_counter()
        whole.diagnostics['sa_calls'] += 1
        self.diagnostic['sa_calls'] += 1
        seeds = [int(np.random.SeedSequence([o.seed, whole.diagnostics['sa_calls'], j]).generate_state(1)[0])
                 for j in range(o.sa_reads)]
        proposals, sampling = anneal(q, o, whole.remaining(), seeds)
        whole.timings['annealing'] += perf_counter()-phase
        sampling.update(block='N', round=k, seed=seeds, neighbourhood=describe(self.current),
                        free_bits=len(q.free), feasibility_cuts_stored=stored, feasibility_cuts_encoded=encoded,
                        anchor=anchor_id, anchor_cost=before, raw_candidates=len(proposals),
                        raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals))
        sampling.pop('raw_bits', None)
        whole.diagnostics['sampling'].append(sampling)
        del q
        outside = []
        for j, x in enumerate(proposals):
            if whole.remaining() <= 0:
                break
            self.evaluate(x, label+'_raw')
            block_x = np.zeros(len(b.obj)); block_x[b.binary] = x
            raw = block_x[b.u]
            record = {'proposal_id': f"{label}:{whole.diagnostics['sa_calls']}:{j}", 'round': k,
                      'neighbourhood': describe(self.current),
                      'raw_outside_changes': int(np.count_nonzero((raw != baseline_u) & ~self.free_u))}
            repaired = whole.prepare(raw, label, record)
            if repaired is not None and whole.remaining() > 0:
                u = repaired[:whole.d.n_units*whole.d.horizon].reshape(raw.shape, order='F')
                record['repair_outside_changes'] = int(np.count_nonzero((u != raw) & ~self.free_u))
                outside.append(record['repair_outside_changes'])
                self.evaluate(repaired[columns], label)
                whole.accept(repaired, label, record)
        self.summarize()
        return dict(sa_call=whole.diagnostics['sa_calls'], neighbourhood=describe(self.current),
                    free_bits=sampling['free_bits'], qubo_bits=sampling['bits'], seconds=sampling['seconds'],
                    before=float(before), after=float(whole.result.objective),
                    improved=bool(whole.result.objective < before), repair_outside_changes=outside)


# 以给定方案为已校验 incumbent 建立全天搜索状态（与 v1.1.0 paper_pilot.start_whole 相同）；不可行则报错。
def start_whole(d, scenarios, o, u, reference, seconds, started=None):
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    started = perf_counter() if started is None else started
    whole = search_core.WholeDay(m, full, o, started+seconds, solver.Result(reference=dict(reference)), started)
    if not whole.accept(pack(d, np.asarray(u, float)), 'initial', {'proposal_id': 'start'}):
        raise RuntimeError('Start schedule is not feasible')
    return whole


# 按概率加权的出力；没有场景调度时返回 None。
def expected_output(result):
    if not result.scenarios:
        return None
    return sum(s.probability*np.asarray(s.p, float) for s in result.scenarios)


# 手写规则 N_rule（不用参考解）：按“每 MWh 平均成本”=（空载费 + 启动费 + 期望燃料费）/ 期望发电量
# 从高到低取前 limit 台已开机组作目标；窗口为目标开机时段前后各扩 1 小时；伴随机组取窗口内至少
# 一小时停机、按 pmax 从小到大（同 pmax 按机组号）的 max_units-1 台。
def rule_neighbourhoods(d, result, limit=4, max_units=MAX_UNITS):
    u = np.asarray(result.u, int)
    starts = np.maximum(u - np.column_stack([d.initial_on, u[:, :-1]]), 0)
    energy = expected_output(result).sum(axis=1)
    fuel = sum(s.probability*(d.cost_a[:, None]*np.asarray(s.p)**2 + d.cost_b[:, None]*np.asarray(s.p))
               for s in result.scenarios).sum(axis=1)
    total = d.cost_c*u.sum(axis=1) + d.startup_cost*starts.sum(axis=1) + fuel
    on = np.flatnonzero(u.sum(axis=1))
    average = {i: total[i]/energy[i] if energy[i] > 1e-9 else np.inf for i in on}
    out = []
    for i in sorted(on, key=lambda i: (-average[i], i))[:limit]:
        hours = np.flatnonzero(u[i])
        start, end = max(0, hours[0]-1), min(d.horizon, hours[-1]+2)
        pool = [j for j in range(d.n_units) if j != i and (u[j, start:end] == 0).any()]
        companions = sorted(pool, key=lambda j: (d.pmax[j], j))[:max_units-1]
        out.append(neighbourhood([i, *companions], start, end))
    return out


# 诊断邻域 N_oracle（用到参考解）：与参考解按同型机组（参数与初始状态全同）比较每小时开机台数，
# 有差异的组的全部机组 × 全天。同型机组互换不改变费用，因此该邻域包含与参考解等价的方案。
UNIT_KEYS = ('pmax', 'pmin', 'pstart', 'pshut', 'min_up', 'min_down', 'ramp_up', 'ramp_down', 'initial_on',
             'initial_duration', 'initial_output', 'cost_a', 'cost_b', 'cost_c', 'startup_cost', 'shutdown_cost')


def unit_groups(d):
    groups = {}
    for i in range(d.n_units):
        groups.setdefault(tuple(float(np.asarray(getattr(d, k))[i]) for k in UNIT_KEYS), []).append(i)
    return list(groups.values())


def oracle_neighbourhood(d, u, reference_u):
    u, reference_u = np.asarray(u, int), np.asarray(reference_u, int)
    units = [i for g in unit_groups(d) if (u[g].sum(axis=0) != reference_u[g].sum(axis=0)).any() for i in g]
    return neighbourhood(units, 0, d.horizon)


# 对照 RM（不用 SA）：邻域内随机区间改动，不限台数。每个提议随机取 1..|units| 台邻域机组，
# 每台在窗口内做一个动作：关掉某段开机区间的一部分，或开一段长度至少 min_up 的区间。
def random_move(d, base_u, n, rng):
    u = np.asarray(base_u, int).copy()
    chosen = rng.choice(len(n.units), size=int(rng.integers(1, len(n.units)+1)), replace=False)
    for k in chosen:
        i = n.units[int(k)]
        row = u[i, n.start:n.end]
        on = np.flatnonzero(row)
        if len(on) and rng.random() < .5:
            a = int(rng.choice(on)); b = a
            while b+1 < len(row) and row[b+1]:
                b += 1
            lo = int(rng.integers(a, b+1)); hi = int(rng.integers(lo, b+1))
            row[lo:hi+1] = 0
        else:
            length = int(min(len(row), max(1, d.min_up[i]) + rng.integers(0, 4)))
            lo = int(rng.integers(0, len(row)-length+1))
            row[lo:lo+length] = 1
        u[i, n.start:n.end] = row
    return u


# 在邻域内连续做随机改动直到 seconds 用完；只评估通过必要条件的新方案（来源 'rm'）。
def random_moves(whole, neighbourhoods, seconds, seed):
    rng = np.random.default_rng(seed)
    started, seen, counts = perf_counter(), set(), dict(proposed=0, duplicate=0, rejected=0, evaluated=0)
    k = 0
    while perf_counter()-started < seconds and whole.remaining() > 0:
        n = neighbourhoods[k % len(neighbourhoods)]; k += 1
        u = random_move(whole.d, whole.result.u, n, rng)
        counts['proposed'] += 1
        x = pack(whole.d, u)
        key = x.astype(np.uint8).tobytes()
        if key in seen:
            counts['duplicate'] += 1
            continue
        seen.add(key)
        known = np.isfinite(whole.fixed)
        reason, _ = necessary(whole.d, whole.scenarios, u)
        if reason is not None or not np.array_equal(x[known], whole.fixed[known]) or not discrete_feasible(whole.full, x):
            counts['rejected'] += 1
            continue
        whole.accept(x, 'rm', {'proposal_id': f'rm:{k}'})
        counts['evaluated'] += 1
    return counts


# UB 降幅按来源拆分：initial、块 SA（sa、pool）、邻域 SA（nsa）、程序候选（operator）、随机改动（rm）、其他。
def decrease_by_source(evaluations):
    groups = dict(initial='initial', initialization_mip='initial', sa='sa', pool='sa', nsa='neighbourhood_sa',
                  operator='operator', rm='random_moves')
    totals = dict(initial=0., sa=0., neighbourhood_sa=0., operator=0., random_moves=0., other=0.)
    for e in evaluations:
        if e['improved'] and np.isfinite(e['incumbent_before']):
            totals[groups.get(e['source'], 'other')] += e['incumbent_before']-e['incumbent']
    return totals


# 固定 u 后全天调度、恢复完整解并独立校验；返回已校验费用（unpack 口径），失败时保留原因。
def evaluate_u(m, full, u, seconds=60.):
    started = perf_counter()
    try:
        r = dispatch(full, pack(m.d, np.asarray(u, dtype=float)), seconds, certified=True)
        if not r.feasible:
            return dict(feasible=False, status=r.status, objective=None, seconds=perf_counter()-started)
        plan = unpack(m, r.point)
        validate(m.d, m.scenarios, plan)
        return dict(feasible=True, status=r.status, objective=plan.objective, commitment_id=commitment_hash(plan),
                    seconds=perf_counter()-started)
    except (RuntimeError, ValueError) as error:
        return dict(feasible=None, status='ERROR', objective=None, message=f'{type(error).__name__}: {error}',
                    seconds=perf_counter()-started)


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


def _solve(g, m, x, anchor_u, seconds):
    x.Start = np.r_[pack(m.d, np.asarray(anchor_u, float)), np.full(len(m.obj)-3*m.d.n_units*m.d.horizon,
                                                                    gp.GRB.UNDEFINED)]
    g.setObjective(m.obj @ x + x @ sparse.diags(m.q) @ x)
    g.setParam('TimeLimit', max(float(seconds), 1e-2))
    started = perf_counter()
    g.optimize()
    out = dict(status=STATUS.get(g.Status, f'STATUS_{g.Status}'), seconds=perf_counter()-started,
               mip_objective=float(g.ObjVal) if g.SolCount else None, bound=None)
    try:
        out['bound'] = float(g.ObjBound)
    except (AttributeError, gp.GurobiError):
        pass
    return out, (np.asarray(x.X) if g.SolCount else None)


# 取整后的 u 独立调度校验。status/seconds 保留 MIP 自己的值，校验结果用 dispatch_status/check_seconds，
# 以免调度 QP 的 OPTIMAL 冒充 MIP 已证最优。proven 要求 MIP 状态为 OPTIMAL。
def checked(out, m, full, u, anchor_u):
    check = evaluate_u(m, full, u)
    out.update(objective=check['objective'], feasible=check['feasible'], dispatch_status=check['status'],
               check_seconds=check['seconds'], commitment_id=check.get('commitment_id'),
               changed_units=int(np.count_nonzero((u != anchor_u).any(axis=1))), proven=out['status'] == 'OPTIMAL')
    return out


# 尺子 C_N：邻域外的 u 固定为 anchor_u，求真实成本最低的方案（取整后独立调度校验）。
def neighbourhood_ceiling(m, full, anchor_u, n, seconds=60.):
    anchor_u = np.asarray(anchor_u, float)
    free_u, _ = free_masks(m.d, n)
    lb, ub = m.lb.copy(), m.ub.copy()
    lb[m.ix.u[~free_u]] = ub[m.ix.u[~free_u]] = anchor_u[~free_u]
    with _model(m, lb, ub) as (g, x):
        out, point = _solve(g, m, x, anchor_u, seconds)
    return out if point is None else checked(out, m, full, np.rint(point[m.ix.u]), anchor_u)


# 尺子：全天任意小时，但至多改动 k 台机组的 u（相对 anchor_u），求真实成本最低的方案。
def unit_budget(m, full, anchor_u, k, seconds=90.):
    anchor_u = np.asarray(anchor_u, float)
    n, T = m.d.n_units, m.d.horizon
    with _model(m) as (g, x):
        z = g.addMVar(n, vtype=gp.GRB.BINARY)
        index = m.ix.u
        rows = sparse.csr_matrix((np.ones(n*T), (np.arange(n*T), index.ravel())), shape=(n*T, len(m.obj)))
        pick = sparse.csr_matrix((np.ones(n*T), (np.arange(n*T), np.repeat(np.arange(n), T))), shape=(n*T, n))
        a = anchor_u.ravel()
        g.addConstr(rows @ x - pick @ z <= a)
        g.addConstr(-(rows @ x) - pick @ z <= -a)
        g.addConstr(z.sum() <= k)
        out, point = _solve(g, m, x, anchor_u, seconds)
    return out if point is None else checked(out, m, full, np.rint(point[m.ix.u]), anchor_u)


# 改进判定：已校验费用严格低于 incumbent·(1 - REL_TOLERANCE)。
def improves(objective, incumbent, rel=REL_TOLERANCE):
    return objective is not None and objective < incumbent - rel*abs(incumbent)
