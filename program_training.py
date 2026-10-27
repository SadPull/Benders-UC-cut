"""RTS24 多场景多轮闭环程序训练：协议常量、训练实例、固定锚点探测、反馈、闭环检查与停止规则。

探测、J 计分、手写对照组与提示模板移植自已归档的 program_synthesis.py / run_program_synthesis.py
（原 IEEE118 生成实验的重建，见 LLM-benders-SUC-archive-20261026.zip）。与旧协议不同：
训练实例保留 3 个负荷场景，程序另见各场景负荷与概率；锚点取 120 秒 SA+R 运行的初始化后、成本降幅过半时与终点；
轮数由预登记的停止规则决定（至少 4 轮、至多 10 轮），每轮反馈覆盖此前全部程序的实测结果。
"""
import json
import random
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

import offline_pilot
from offline_pilot import Program, ProgramError, action, candidate_u, necessary
from offline_worker import intervals
from suc_benders import solver
from suc_benders.benders import discrete_feasible
from suc_benders.config import Options
from suc_benders.model import model, partition
from suc_benders.search import WholeDay
from suc_benders.validation import pack, commitment, commitment_hash, validate

CONTROLS = ['R', 'H0', 'H1', 'H2', 'H3']
OP_STREAM = 202610231
# (0,0) 即测试实例 rts24-base，不能用于训练或验证。
SPLITS = {'train': [(-.04, .02), (.04, -.02), (.03, .03)],
          'validation': [(-.02, -.02), (.02, .02)]}
# 锚点来自无 LLM 的 SA+R 运行（只用 SA 时初始化之后不再改进）：初始化之后、成本降幅首次达到全程一半时、终点。
ANCHOR_ARM, ANCHOR_SECONDS, PROBE_SECONDS = 'SA+R', 120., 15
ROLES = ('initial', 'half', 'final')
SEEDS = dict(anchors=103, train=211, validation=307)
MIN_ROUNDS, MAX_ROUNDS, PATIENCE = 4, 10, 2
# 第 r 轮算改进：B_r >= B_{r-1} + max(DELTA_ABS, DELTA_REL*|B_{r-1}|)，J 以百分点计。
DELTA_ABS, DELTA_REL = .10, .03
# 同一轮因传输、CLI 或超时失败时的最多尝试次数；不计入轮数。
ATTEMPTS = 3
# 反馈附完整源码的程序：训练均分前 SOURCE_TOP 名加上一轮的全部程序。
SOURCE_TOP = 4
MIN_TEST_DISTANCE = .005
REJECTIONS = ('timeline', 'capacity', 'minimum_output', 'ramp_reachability', 'model_binary_constraints')
GATE_J = .05


def options(seed, seconds):
    return Options(seed=seed, time_limit=seconds, block_hours=12, max_rounds=1000000,
                   sa_reads=8, sa_sweeps=16, sa_temperature=4., sa_cooling=.95,
                   sa_cutoff=.05, llm_enabled=False, verbose=False)


def load_profile(base, a, b):
    phase = np.arange(24) * np.pi / 23
    load = np.asarray(base) * (1 + a*np.sin(phase) + b*np.sin(2*phase))
    load[[0, -1]] = np.asarray(base)[[0, -1]]
    return load


def variants():
    return {f'{split}{k}': dict(split=split, a=a, b=b, stream_index=n)
            for n, (split, k, a, b) in enumerate(((s, k, a, b) for s, pairs in SPLITS.items()
                                                   for k, (a, b) in enumerate(pairs, 1)), 1)}


def scenario_loads(scenarios):
    return np.asarray([s.load for s in scenarios], float)


# 每个场景逐时比较，取最大相对差；训练与验证变体对每个测试实例都必须至少相差 MIN_TEST_DISTANCE。
def test_distance(loads, test):
    return float(np.max(np.abs(loads-test)/np.abs(test)))


# improvements 为 (秒, 方案) 的严格改进序列；未改进时三个锚点相同（另作标注，照常探测）。
def choose_anchors(initial, improvements):
    final = improvements[-1] if improvements else initial
    target = initial[1].objective-(initial[1].objective-final[1].objective)/2
    half = next((x for x in improvements if x[1].objective <= target), final)
    return dict(initial=initial, half=half, final=final)


def weighted_dispatch(scenarios, plan):
    return sum(s.probability*np.asarray(sp.p, float) for s, sp in zip(scenarios, plan.scenarios))


def estimated_saving(d, base, p, u):
    expense = d.cost_a[:, None]*p*p + d.cost_b[:, None]*p + d.cost_c[:, None]*base
    minimum = (d.cost_a*d.pmin*d.pmin+d.cost_b*d.pmin+d.cost_c)[:, None]
    return float(np.sum(expense*(base > u))-np.sum(minimum*(u > base)))


# 手写对照 R（即方法 SA+R 的随机区间算子）、H0–H3：每次至多 64 个不同动作，必要检查后 H 按估计节省排序、R 随机返回。出力按场景概率加权。
def handwritten(d, scenarios, plan, identifier, seed, deadline):
    rng = random.Random(seed)
    base = plan.u.astype(int).tolist()
    segments = [(i, a, b) for i, row in enumerate(base) for a, b in intervals(row)]
    p = weighted_dispatch(scenarios, plan)
    seen, records, survivors = set(), [], []
    for _ in range(1024):
        if len(seen) >= 64 or perf_counter() >= deadline:
            break
        kind = rng.choice(CONTROLS[1:]) if identifier == 'R' else identifier
        proposal = action(base, kind, rng, segments)
        if proposal is None:
            continue
        key = json.dumps(proposal['changes'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        u = candidate_u(plan.u, proposal)
        reason, detail = necessary(d, scenarios, u)
        estimate = estimated_saving(d, plan.u, p, u)
        records.append(dict(kind=kind, **proposal, rejection=reason, detail=detail, estimate=estimate))
        if reason is None:
            survivors.append((estimate, key, proposal))
    if identifier == 'R':
        rng.shuffle(survivors)
    else:
        survivors.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in survivors[:4]], records


def stream_seed(seed, variant, anchor, call):
    return int(np.random.SeedSequence([seed, OP_STREAM, variant, anchor, call]).generate_state(1)[0])


def snapshot(result):
    return NS(objective=float(result.objective), **{n: getattr(result, n).copy() for n in ('u', 'v', 'w')},
              scenarios=[NS(**{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in vars(s).items()})
                         for s in result.scenarios])


def restore(value):
    return NS(objective=value['objective'], **{n: np.asarray(value[n]) for n in ('u', 'v', 'w')},
              scenarios=[NS(**dict(s, p=np.asarray(s['p']), r=np.asarray(s['r']))) for s in value['scenarios']])


class Operator:
    """固定锚点探测的操作程序：每次调用都从同一锚点出发，候选只经必要检查与全天调度、独立校验。"""
    def __init__(self, whole, identifier, source=None, variant=0, anchor=0):
        self.whole, self.identifier = whole, identifier
        self.source, self.variant, self.anchor = source, variant, anchor
        self.program, self.calls, self.empty = None, 0, 0
        self.seen = {pack(whole.d, whole.result.u).astype(np.uint8).tobytes()}
        whole.diagnostics.update(operator_calls=[], operator_candidates=[], operator_generation_seconds=0.)

    def step(self, plan):
        whole = self.whole
        if whole.remaining() <= 0:
            raise TimeoutError
        self.calls += 1
        seed = stream_seed(whole.o.seed, self.variant, self.anchor, self.calls)
        call = dict(call=self.calls, seed=seed, anchor_hash=commitment_hash(plan),
                    started_seconds=perf_counter()-whole.started, actions=[])
        whole.diagnostics['operator_calls'].append(call)
        before = whole.result.objective
        phase = perf_counter()
        try:
            if self.source is None:
                candidates, call['actions'] = handwritten(whole.d, whole.scenarios, plan,
                    self.identifier, seed, whole.deadline)
            else:
                if self.program is None:
                    self.program = Program(self.source, whole.deadline)
                candidates = self.program.propose(offline_pilot.context(whole.d, whole.scenarios, plan, True),
                                                  seed, whole.deadline)
            call['generation_seconds'] = perf_counter()-phase
            whole.diagnostics['operator_generation_seconds'] += call['generation_seconds']
            if whole.remaining() <= 0:
                raise TimeoutError('Late generation result discarded')
            if not isinstance(candidates, list) or len(candidates) > 4:
                raise ProgramError('Interface requires a list of at most four candidates')
            call['returned'] = len(candidates)
            new = 0
            for k, candidate in enumerate(candidates):
                record = dict(proposal_id=f'operator:{self.calls}:{k}', call=self.calls,
                    candidate=candidate, format_valid=False, necessary_pass=False,
                    started_seconds=perf_counter()-whole.started)
                whole.diagnostics['operator_candidates'].append(record)
                phase = perf_counter()
                try:
                    u = candidate_u(plan.u, candidate)
                except (ValueError, TypeError, KeyError) as error:
                    record.update(outcome='format_invalid', message=str(error))
                    if self.source is not None:
                        raise ProgramError('Invalid output: '+str(error)) from error
                    continue
                record['format_valid'] = True
                x = pack(whole.d, u)
                key = x.astype(np.uint8).tobytes()
                record['commitment'] = commitment_hash(commitment(whole.d, x))
                if key in self.seen:
                    record.update(outcome='duplicate', duplicate=True)
                    continue
                self.seen.add(key); new += 1
                reason, detail = necessary(whole.d, whole.scenarios, u)
                known = np.isfinite(whole.fixed)
                if reason is None and (not np.array_equal(x[known], whole.fixed[known])
                        or not discrete_feasible(whole.full, x)):
                    reason = 'model_binary_constraints'
                record.update(necessary_pass=reason is None, precheck_seconds=perf_counter()-phase,
                              rejection=reason, detail=detail)
                if whole.remaining() <= 0:
                    record['outcome'] = 'deadline'
                    break
                if reason is not None:
                    record['outcome'] = 'necessary_rejected'
                    continue
                # accept 本身只做调度和独立校验；不经过 prepare/repair。
                whole.accept(x, 'operator', record)
                record['ended_seconds'] = perf_counter()-whole.started
            self.empty = self.empty+1 if new == 0 else 0
            call.update(new_candidates=new, status='OK')
        except ProgramError as error:
            call.update(status='PROGRAM_INVALID', error=str(error))
            raise
        finally:
            call.update(ended_seconds=perf_counter()-whole.started, improved=whole.result.objective < before)
            whole.emit('operator')

    def close(self):
        if self.program is not None:
            self.program.close()


def probe(d, scenarios, anchor, identifier, source, seed, variant, anchor_index, progress=None, started_at=None):
    started = perf_counter() if started_at is None else started_at
    result = solver.Result()
    for name in ('u', 'v', 'w', 'scenarios', 'objective'):
        setattr(result, name, getattr(anchor, name))
    result.validation = validate(d, scenarios, anchor)
    m = model(d, scenarios)
    whole = WholeDay(m, partition(m, d.horizon)[0][0], options(seed, PROBE_SECONDS),
                     started+PROBE_SECONDS-solver.DEADLINE_MARGIN_SECONDS, result, started, progress)
    operator = Operator(whole, identifier, source, variant, anchor_index)
    try:
        while whole.remaining() > 0:
            operator.step(anchor)
        result.status = 'TIME_LIMIT'
    except TimeoutError:
        result.status = 'TIME_LIMIT'
    except Exception as error:
        result.status = 'PROGRAM_INVALID' if isinstance(error, ProgramError) else 'PROBE_FAILED'
        result.message = f'{type(error).__name__}: {error}'
    finally:
        operator.close(); whole.advice.close()
    result.validation = validate(d, scenarios, result)
    result.elapsed_seconds = perf_counter()-started
    points = [(0., anchor.objective)]
    for row in result.diagnostics['evaluations']:
        if row['improved'] and row['elapsed_seconds'] <= PROBE_SECONDS-solver.DEADLINE_MARGIN_SECONDS:
            points.append((row['elapsed_seconds'], row['incumbent']))
    return result, points


def improvement_score(points, anchor, seconds=PROBE_SECONDS):
    return sum(((points[k+1][0] if k+1 < len(points) else seconds)-t)*100*(anchor-c)/abs(anchor)
               for k, (t, c) in enumerate(points))/seconds


def statistics(diagnostics):
    rows = diagnostics.get('operator_candidates', [])
    calls = diagnostics.get('operator_calls', [])
    return dict(calls=len(calls), returned_candidates=len(rows),
        format_valid=sum(r['format_valid'] for r in rows), duplicate=sum(r.get('duplicate', False) for r in rows),
        necessary_pass=sum(r['necessary_pass'] for r in rows),
        rejections={k: sum(r.get('rejection') == k for r in rows) for k in REJECTIONS},
        dispatch_feasible=sum(r.get('feasible', False) for r in rows),
        improvements=sum(r.get('incumbent_decrease', 0) > 0 for r in rows),
        generation_seconds=diagnostics.get('operator_generation_seconds', 0.))


def valid(row):
    return (row['status'] == 'TIME_LIMIT' and row['feasible']
            and row['elapsed_seconds'] <= row['budget_seconds'] and row.get('llm_calls', 0) == 0)


def mean_j(rows, generator, stage, count):
    selected = [r for r in rows if r['generator'] == generator and r['stage'] == stage]
    return float(np.mean([r['J'] for r in selected])) if len(selected) == count and all(map(valid, selected)) else None


def counts():
    return dict(train=len(SPLITS['train'])*len(ROLES), validation=len(SPLITS['validation'])*len(ROLES))


def _sum_statistics(rows):
    total = {}
    for r in rows:
        for k, v in r['statistics'].items():
            if isinstance(v, dict):
                total.setdefault(k, {})
                for name, n in v.items():
                    total[k][name] = total[k].get(name, 0)+n
            else:
                total[k] = total.get(k, 0)+v
    return total


# 第 r 轮的输入：对照组与此前全部程序的训练测量（不含任何验证数字）；训练均分前 SOURCE_TOP 名与上一轮程序附完整源码。
def feedback(rows, programs, round_index, sources):
    train = [r for r in rows if r['stage'] == 'train']
    earlier = {i: p for i, p in programs.items() if p['round'] < round_index}
    scored = {i: mean_j(train, i, 'train', counts()['train']) for i in earlier}
    ranked = sorted((i for i, v in scored.items() if v is not None), key=lambda i: (-scored[i], i))
    with_source = set(ranked[:SOURCE_TOP]) | {i for i, p in earlier.items() if p['round'] == round_index-1}
    entries = []
    for identifier in CONTROLS+sorted(earlier, key=lambda i: (earlier[i]['round'], i)):
        mine = [r for r in train if r['generator'] == identifier]
        entry = dict(generator=identifier, kind='control' if identifier in CONTROLS else 'program',
                     train_mean_J=mean_j(train, identifier, 'train', counts()['train']),
                     measurements=[{k: r[k] for k in ('variant', 'anchor', 'status', 'J')} |
                                   ({'message': r['message']} if r['status'] != 'TIME_LIMIT' else {}) for r in mine],
                     statistics=_sum_statistics(mine) if mine else None)
        if identifier in earlier:
            p = earlier[identifier]
            entry.update(round=p['round'], reason=p.get('reason'), status=p['status'], error=p.get('error'),
                         code=sources(identifier) if identifier in with_source and p.get('sha256') else None)
        entries.append(entry)
    return entries


# 发出第 r 轮请求前的闭环断言：此前每个程序都在反馈中出现；已评估的程序带满训练测量，未评估的写明原因。
def closed_loop_check(round_index, entries, programs, rows):
    listed = {e['generator']: e for e in entries}
    record = []
    for identifier, p in sorted(programs.items()):
        if p['round'] >= round_index:
            continue
        entry = listed.get(identifier)
        if entry is None:
            raise RuntimeError(f'Feedback for round {round_index} omits {identifier}')
        measured = [r['folder'] for r in rows if r['generator'] == identifier and r['stage'] == 'train']
        if p['status'] == 'EVALUATED':
            if len(entry['measurements']) != counts()['train'] or len(measured) != counts()['train']:
                raise RuntimeError(f'Feedback for round {round_index} lacks training measurements of {identifier}')
        elif not (p.get('error') or p['status'] in ('DUPLICATE',)):
            raise RuntimeError(f'Unevaluated program {identifier} has no recorded reason')
        record.append(dict(program=identifier, round=p['round'], sha256=p.get('sha256'), status=p['status'],
                           train_rows=measured, source_included=entry.get('code') is not None))
    if any(m.get('variant', '').startswith('validation') for e in entries for m in e['measurements']):
        raise RuntimeError('Validation measurements must not reach the LLM')
    return record


def improved(previous, current):
    if current is None:
        return False
    if previous is None:
        return True
    return current >= previous+max(DELTA_ABS, DELTA_REL*abs(previous))


# best[k] 为完成第 k+1 轮后的 B（截至该轮所有有效程序的最佳验证均分；尚无有效程序时为 None）。
def stop_decision(best):
    r = len(best)
    flags = [improved(best[k-1] if k else None, best[k]) for k in range(r)]
    if r >= MAX_ROUNDS:
        reason = 'MAX_ROUNDS'
    elif r >= MIN_ROUNDS and not any(flags[-PATIENCE:]):
        reason = 'CONVERGED'
    else:
        reason = 'CONTINUE'
    return dict(stop=reason != 'CONTINUE', reason=reason, rounds=r, improvements=flags, best=best)


def best_validation(rows, programs, up_to_round):
    values = [v for i, p in programs.items() if p['round'] <= up_to_round and p['status'] == 'EVALUATED'
              for v in [mean_j(rows, i, 'validation', counts()['validation'])] if v is not None]
    return max(values) if values else None


# final：全部轮次中验证均分最高的程序（同分取较早轮次）；early：第 1 轮验证均分最高的程序。
def select(rows, programs):
    def best(ids):
        scored = [(mean_j(rows, i, 'validation', counts()['validation']), programs[i]['round'], i) for i in ids]
        scored = [s for s in scored if s[0] is not None]
        return min(scored, key=lambda s: (-s[0], s[1], s[2]))[2] if scored else None
    evaluated = [i for i, p in programs.items() if p['status'] == 'EVALUATED']
    return dict(final=best(evaluated), early=best([i for i in evaluated if programs[i]['round'] == 1]))


def gate(rows, chosen):
    if chosen is None:
        return dict(passed=False, reason='No valid program')
    by_variant = {v: float(np.mean([r['J'] for r in rows if r['generator'] == chosen and r['variant'] == v]))
                  for v in (f'validation{k}' for k in range(1, len(SPLITS['validation'])+1))}
    mean, control = (mean_j(rows, i, 'validation', counts()['validation']) for i in (chosen, 'R'))
    passed = min(by_variant.values()) >= GATE_J and mean is not None and control is not None and mean > control
    return dict(passed=bool(passed), chosen=chosen, mean_J_by_variant=by_variant, validation_mean=mean,
                R_validation_mean=control, rule=f'Each validation variant mean J >= {GATE_J} and validation mean > R')
