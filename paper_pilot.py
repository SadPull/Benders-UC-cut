"""论文实验：多场景 IEEE118 与 RTS24 上 SA 与 SA+L118 的对比、QUBO 比特数记录与案例。"""
# 试探脚本，不属于生产代码。运行沿用 offline_pilot.run_trial；参考解只用来计算差距，从不注入搜索。
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

import offline_pilot
from offline_pilot import EMPTY_LIMIT, L118, Operator
from suc_benders import search as search_core, solver
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.validation import commitment, pack

ROOT = Path(__file__).resolve().parent
INSTANCES = dict(ieee118=ROOT/'data', rcuc20=ROOT/'instances/rcuc20', rts24=ROOT/'instances/rts24',
                 ieee118s10=ROOT/'instances/ieee118s10')
ARMS = ('SA', 'SA+L118')
# 参考解未证明最优时，Gurobi 自报 gap 不超过此值（百分比）才可当尺子。
RULER_GAP = .05
# 判据：SA+L118 终点差距严格更小的种子数下限，与 AUC 降幅中位数下限。
WINS_NEEDED, AUC_REDUCTION = 4, .10
# 资源估算假设 θ 为 12 位；每条割另按 1 或 12 位松弛寄存器分别计数。
THETA_BITS = 12
# 案例：从停滞点起至多调用 L118 的次数。
CASE_CALLS = 50


def instance(name):
    folder = INSTANCES[name]
    return load_data(folder/'units.json', folder/'stochastic.json')


# 与 run_experiment.py 的纯 SA 设置一致：12 h 分块、16 sweeps、LLM 关闭，其余取默认值。
def options(name, seed, seconds):
    folder = INSTANCES[name]
    return load_options(dict(data_file=str(folder/'units.json'), scenario_file=str(folder/'stochastic.json'),
                             seed=seed, time_limit=seconds, block_hours=12, sa_sweeps=16, max_rounds=10**6,
                             llm_enabled=False, verbose=False))


# 可当尺子的参考解：校验可行，且证明最优或 Gurobi 自报 gap 不超过 RULER_GAP。
def ruler(reference):
    valid = (reference.get('validation') or {}).get('feasible') and reference.get('objective') is not None
    proven = reference.get('status') == 'OPTIMAL' or (
        reference.get('status') == 'TIME_LIMIT' and (reference.get('gurobi_gap_percent') or 1e9) <= RULER_GAP)
    if not (valid and proven):
        raise RuntimeError(f"reference unusable: status={reference.get('status')} "
                           f"gap={reference.get('gurobi_gap_percent')}")
    return dict(status=reference['status'], objective=reference['objective'], validation=reference['validation'])


# 只读记录器：每次建立块搜索 QUBO 时记下本文编码的实际位数与割数，行为与生产版完全相同。
class CountingSearch(search_core.BlockSearch):
    log = None

    def build_qubo(self, anchor, boundary):
        q, stored, encoded = super().build_qubo(anchor, boundary)
        if CountingSearch.log is not None:
            optimality = sum(c.optimality for c in self.cuts)
            CountingSearch.log.append(dict(
                t=perf_counter()-self.whole.started, block=self.index, master=self.diagnostic['sa_calls']+1,
                free=len(q.free), ours=int(q.Q.shape[0]), stored_optimality=optimality,
                stored_feasibility=len(self.cuts)-optimality, feasibility_encoded=encoded,
                total_optimality=self.cut_totals['optimality'], total_feasibility=self.cut_totals['feasibility']))
        return q, stored, encoded


@contextmanager
def counted(log):
    original, CountingSearch.log = search_core.BlockSearch, log
    search_core.BlockSearch = CountingSearch
    try:
        yield log
    finally:
        search_core.BlockSearch, CountingSearch.log = original, None


# 给定 θ 和每条割寄存器位数后的资源估算，不是无条件下界，也不主张与费用代理等价：
# same_cuts 编码本文同一批已存割；single_cut 为经典单割 Benders，每次主问题求解只多一条割。
def standard_bits(row, per_cut):
    same = row['stored_optimality'] + row['stored_feasibility']
    return dict(same_cuts=row['free'] + THETA_BITS + per_cut*same,
                single_cut=row['free'] + THETA_BITS + per_cut*(row['master']-1))


# 一次冷启动运行（offline_pilot.run_trial），附带 QUBO 位数记录。data、o 只供测试。
def counted_trial(name, arm, seed, seconds, reference, rounds=None, burst_calls=None, data=None, o=None):
    data = instance(name) if data is None else data
    o = options(name, seed, seconds) if o is None else o
    with counted([]) as log:
        row = offline_pilot.run_trial(name, arm, seed, seconds, reference, rounds, burst_calls, data=data, o=o)
    return dict(row, qubo_sizes=log)


# 按种子配对 SA 与 SA+L118；要求每个预定种子都有完整配对。gap 缺失（不可行）按无穷大计。
def evaluate(runs, name, seeds):
    pairs = {}
    for r in runs:
        if r['instance'] == name:
            pairs.setdefault(r['seed'], {})[r['arm']] = r
    complete = [pairs[s] for s in seeds if set(ARMS) <= set(pairs.get(s, {}))]
    gap = lambda r: r['gap_percent'] if r['gap_percent'] is not None else np.inf
    wins = sum(gap(p['SA+L118']) < gap(p['SA']) - 1e-9 for p in complete)
    reductions = [1 - p['SA+L118']['auc']/p['SA']['auc'] for p in complete if p['SA']['auc']]
    median = float(np.median(reductions)) if reductions else None
    passed = len(complete) == len(seeds) and wins >= WINS_NEEDED and median is not None and median >= AUC_REDUCTION
    return dict(pairs=len(complete), wins=wins, auc_reduction=median, passed=bool(passed))


# SA+L118 首次达到 SA 终点成本的时间；达不到返回 None。
def time_to_reach(trajectory, objective):
    return next((t for t, cost in trajectory if cost <= objective + 1e-6), None)


# 以给定方案为已校验的当前方案建立全天搜索状态；方案不可行则报错。
def start_whole(d, scenarios, o, u, reference, seconds):
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    started = perf_counter()
    whole = search_core.WholeDay(m, full, o, started+seconds, solver.Result(reference=dict(reference)), started)
    if not whole.accept(pack(d, np.asarray(u, float)), 'initial', {'proposal_id': 'start'}):
        raise RuntimeError('Start schedule is not feasible')
    return whole


# 固定当前方案：被接受的更好方案只记录、不替换（只作用于这个实例）。
def hold(whole):
    whole._adopt_incumbent = lambda r, plan, report: None


# 案例：固定在停滞方案上逐次调用 L118（当前方案不切换），取按候选顺序第一个改进全天成本的方案。
def case_study(name, seed, start_u, reference, max_calls=CASE_CALLS, seconds=600.):
    d, scenarios = instance(name)
    o = options(name, seed, seconds)
    whole = start_whole(d, scenarios, o, start_u, reference, seconds)
    hold(whole)
    base_objective, base_u = whole.result.objective, whole.result.u.copy()
    found, accept = [], whole.accept

    def watched(x, source, record=None):
        n = len(whole.diagnostics['evaluations'])
        feasible = accept(x, source, record)
        improved = [e for e in whole.diagnostics['evaluations'][n:] if e['improved']]
        if improved and not found:
            found.append((np.asarray(x).copy(), improved[0], dict(record or {})))
        return feasible

    whole.accept = watched
    operator = Operator(whole, seed, L118.read_text(encoding='utf-8'))
    try:
        while not found and operator.calls < max_calls and operator.empty < EMPTY_LIMIT:
            operator.step()
    finally:
        operator.close()
    price = sum(s.probability*ds.balance_dual for s, ds in zip(scenarios, whole.dispatches))
    load = sum(s.probability*np.asarray(s.load, float) for s in scenarios)
    row = dict(instance=name, seed=seed, calls=operator.calls, outcomes=operator.outcomes, found=bool(found),
               base_objective=float(base_objective), base_u=base_u.astype(int).tolist(),
               expected_load=load.tolist(), expected_price=np.asarray(price).tolist())
    if found:
        x, entry, record = found[0]
        u = commitment(d, x).u
        changed = np.flatnonzero((u != base_u).any(axis=1))
        row.update(proposal_id=record.get('proposal_id'), objective=float(entry['objective']),
                   delta=float(entry['objective']-base_objective), u=u.astype(int).tolist(),
                   changed_units=(changed+1).tolist())
    return row
