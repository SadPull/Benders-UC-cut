"""离线 LLM 程序：沙箱调用、候选检查，以及各方法（SA、SA+R、SA+L118、SA+L24r2、SA+L24）的一次冷启动运行。"""
# 在线不调用 LLM；L118、L24、L24r2 是事先写好的冻结程序（programs/），在隔离子进程中运行。参考最优解只用来计算差距，不注入搜索。
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
from time import perf_counter

import numpy as np

from suc_benders import search as search_core, solver
from suc_benders.benders import discrete_feasible
from suc_benders.data import jsonable
from suc_benders.llm import _apply_changes
from suc_benders.model import partition, priced, search_multipliers
from suc_benders.validation import capacity, commitment, pack, reachable_shortfall, transition_violations
from offline_worker import check_source, intervals, make_candidate

ROOT = Path(__file__).resolve().parent
# 程序按生成时所用的系统命名：L118 在 IEEE118 上生成；L24、L24r2 是 RTS24 多场景多轮闭环训练
# （v1.2.0，训练脚本见 v1.3.0 Release 的 experiment-source）的终选程序与第 2 轮最佳程序。
L118 = ROOT/'programs'/'L118.py'
L24 = ROOT/'programs'/'L24.py'
L24r2 = ROOT/'programs'/'L24r2.py'
PROGRAMS = {'SA+L118': L118, 'SA+L24r2': L24r2, 'SA+L24': L24}
# 这些组的程序在训练时看到各场景负荷与概率，测试时同样提供。
SCENARIO_ARMS = ('SA+L24r2', 'SA+L24')
# SA 只做 SA 轮次；SA+R 用随机区间算子 R（不调用 LLM）；其余用对应程序。
ARMS = ('SA', 'SA+R', 'SA+L118', 'SA+L24r2', 'SA+L24')
OP_STREAM = 202610231
CONTEXT_FIELDS = ('pmin', 'pmax', 'pstart', 'pshut', 'ramp_up', 'ramp_down', 'min_up',
                  'min_down', 'initial_on', 'initial_duration', 'initial_output',
                  'cost_a', 'cost_b', 'cost_c', 'startup_cost', 'shutdown_cost')
# 连续这么多次调用没有新候选，操作程序的本次运行结束。
EMPTY_LIMIT = 5


class ProgramError(RuntimeError):
    pass


class Program:
    """每次运行独立子进程；每次调用重新创建命名空间，0.5 秒硬截止。"""
    def __init__(self, source, deadline):
        check_source(source)
        env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP') if k in os.environ}
        self.process = subprocess.Popen([sys.executable, '-I', '-S', str(ROOT/'offline_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.messages = queue.Queue()

        def read():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            response = self.exchange({'source': source}, min(5., deadline-perf_counter()))
            if not response.get('ready'):
                raise ProgramError('Worker failed to initialize')
        except Exception:
            self.close()
            raise

    def exchange(self, request, seconds):
        if seconds <= 0:
            raise TimeoutError('Online deadline')
        deadline = perf_counter()+seconds
        self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
        self.process.stdin.flush()
        try:
            message = self.messages.get(timeout=max(0., deadline-perf_counter()))
        except queue.Empty:
            self.close()
            raise ProgramError('GENERATION_TIMEOUT') from None
        if perf_counter() > deadline:
            self.close()
            raise ProgramError('GENERATION_TIMEOUT')
        if message is None:
            raise ProgramError('Worker exited without a response')
        value = json.loads(message)
        if 'error' in value:
            raise ProgramError(value['error'])
        return value

    def propose(self, context, seed, deadline):
        remaining = deadline-perf_counter()
        if remaining <= 0:
            raise TimeoutError('Online deadline')
        try:
            return self.exchange({'context': context, 'seed': seed}, min(.5, remaining))['candidates']
        except ProgramError:
            if perf_counter() >= deadline:
                raise TimeoutError('Online deadline') from None
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=2)
        self.reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


# 负荷与出力按场景概率加权（单场景时与原值逐位相同）；多场景的可行性仍由 necessary() 和全天调度把关。
# scenario_fields 另给出各场景负荷与概率；SA+L118 与 SA+R 的调用不带这两个字段。
def context(d, scenarios, plan, scenario_fields=False):
    load = sum(s.probability*np.asarray(s.load, float) for s in scenarios)
    p = sum(s.probability*np.asarray(sp.p, float) for s, sp in zip(scenarios, plan.scenarios))
    extra = dict(scenario_loads=[s.load for s in scenarios],
                 scenario_probabilities=[s.probability for s in scenarios]) if scenario_fields else {}
    return jsonable(dict(n_units=d.n_units, horizon=d.horizon,
        **{k: getattr(d, k) for k in CONTEXT_FIELDS}, load=load,
        reserve_fraction=d.reserve_fraction, u=plan.u, p=p, **extra))


def candidate_u(base_u, candidate):
    if not isinstance(candidate, dict) or set(candidate)-{'name', 'reason', 'changes'}:
        raise ValueError('Candidate must contain changes and optional name/reason')
    changes = candidate.get('changes')
    if not isinstance(changes, list) or not 1 <= len(changes) <= 4:
        raise ValueError('Candidate must edit between one and four units')
    return _apply_changes(base_u, changes, *base_u.shape)


def necessary(d, scenarios, u):
    violations = transition_violations(d, u)
    if violations:
        return 'timeline', violations
    z = commitment(d, pack(d, u))
    load = np.asarray([s.load for s in scenarios])
    if (capacity(d, z.u, z.v, z.w).sum(axis=0) < (1+d.reserve_fraction)*load.max(axis=0)-1e-6).any():
        return 'capacity', []
    if (np.sum(d.pmin[:, None]*u, axis=0) > load.min(axis=0)+1e-6).any():
        return 'minimum_output', []
    shortfall = reachable_shortfall(d, scenarios, u)
    return ('ramp_reachability', shortfall) if shortfall else (None, [])


def action(base_u, kind, rng, segments):
    updates = {}
    if not segments:
        return None
    i, a, b = rng.choice(segments)
    row = list(base_u[i])
    if kind == 'H0':
        row[a:b] = [0]*(b-a)
        updates[i] = row
    elif kind == 'H1':
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        start, end = (a+delta, b) if rng.randrange(2) == 0 else (a, b+delta)
        if not 0 <= start < end <= len(row):
            return None
        row[a:b] = [0]*(b-a)
        row[start:end] = [1]*(end-start)
        updates[i] = row
    elif kind == 'H2':
        length = min(rng.choice([1, 2, 3, b-a]), b-a)
        start, end = (a, a+length) if rng.randrange(2) == 0 else (b-length, b)
        j = rng.randrange(len(base_u)-1)
        j += j >= i
        other = list(base_u[j])
        if all(other[start:end]):
            return None
        row[start:end], other[start:end] = [0]*length, [1]*length
        updates = {i: row, j: other}
    else:
        units = sorted({v[0] for v in segments})
        if len(units) < 2:
            return None
        for i in rng.sample(units, rng.randint(2, min(4, len(units)))):
            _, a, b = rng.choice([v for v in segments if v[0] == i])
            length = min(rng.randint(1, 3), b-a)
            row = list(base_u[i]); row[b-length:b] = [0]*length
            updates[i] = row
    candidate = make_candidate(base_u, updates)
    return candidate if candidate['changes'] else None


# 随机区间算子 R（方法 SA+R）：同样的四类区间动作、64 个不同动作、1024 次尝试，不按本轮结果调参。
def random_operator(d, scenarios, plan, seed, deadline):
    rng = random.Random(seed)
    base_u = plan.u.astype(int).tolist()
    segments = [(i, a, b) for i, row in enumerate(base_u) for a, b in intervals(row)]
    seen, survivors = set(), []
    for _ in range(1024):
        if len(seen) >= 64 or perf_counter() >= deadline:
            break
        proposal = action(base_u, rng.choice(('H0', 'H1', 'H2', 'H3')), rng, segments)
        if proposal is None:
            continue
        key = json.dumps(proposal['changes'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if necessary(d, scenarios, candidate_u(plan.u, proposal))[0] is None:
            survivors.append(proposal)
    rng.shuffle(survivors)
    return survivors[:4]


class Operator:
    def __init__(self, whole, seed, source, scenario_fields=False):
        self.whole, self.seed, self.source = whole, seed, source
        self.scenario_fields = scenario_fields
        self.program, self.calls, self.empty, self.seconds = None, 0, 0, 0.
        self.generation_seconds = 0.
        self.seen = {pack(whole.d, whole.result.u).astype(np.uint8).tobytes()}
        self.outcomes = {}

    def step(self):
        whole = self.whole
        if whole.remaining() <= 0:
            raise TimeoutError
        started = perf_counter()
        self.calls += 1
        seed = int(np.random.SeedSequence([self.seed, OP_STREAM, self.calls]).generate_state(1)[0])
        base_u = whole.result.u.copy()
        try:
            generation_started = perf_counter()
            try:
                if self.source is None:
                    candidates = random_operator(whole.d, whole.scenarios, whole.result, seed, whole.deadline)
                else:
                    if self.program is None:
                        self.program = Program(self.source, whole.deadline)
                    candidates = self.program.propose(context(whole.d, whole.scenarios, whole.result,
                                                              self.scenario_fields), seed, whole.deadline)
            finally:
                spent = perf_counter()-generation_started
                self.generation_seconds += spent
                whole.timings['operator_generation'] = whole.timings.get('operator_generation', 0.) + spent
            if whole.remaining() <= 0:
                raise TimeoutError('Late generation result discarded')
            if not isinstance(candidates, list) or len(candidates) > 4:
                raise ProgramError('Interface requires a list of at most four candidates')
            new = 0
            for k, candidate in enumerate(candidates):
                record = {'proposal_id': f'operator:{self.calls}:{k}'}
                try:
                    u = candidate_u(base_u, candidate)
                except (ValueError, TypeError, KeyError) as error:
                    raise ProgramError('Invalid output: '+str(error)) from error
                x = pack(whole.d, u)
                key = x.astype(np.uint8).tobytes()
                if key in self.seen:
                    self.count('duplicate')
                    continue
                self.seen.add(key)
                new += 1
                reason, _ = necessary(whole.d, whole.scenarios, u)
                known = np.isfinite(whole.fixed)
                if reason is None and (not np.array_equal(x[known], whole.fixed[known])
                                       or not discrete_feasible(whole.full, x)):
                    reason = 'model_binary_constraints'
                if reason is not None:
                    self.count('necessary_rejected')
                    continue
                if whole.remaining() <= 0:
                    break
                whole.accept(x, 'operator', record)
                self.count(record.get('outcome', 'none'))
            self.empty = self.empty+1 if new == 0 else 0
        finally:
            self.seconds += perf_counter()-started

    def count(self, outcome):
        self.outcomes[outcome] = self.outcomes.get(outcome, 0)+1

    def close(self):
        if self.program is not None:
            self.program.close()


# 交替规则：操作程序连续运行到用掉 budget 秒（与刚完成的一整轮 SA 相同），或连续 EMPTY_LIMIT 次调用没有新候选。
# calls 只供测试固定调用次数以保证可复现。返回本次用时。
def burst(operator, budget, calls=None):
    started = perf_counter()
    operator.empty = 0
    done = 0
    while operator.empty < EMPTY_LIMIT and (perf_counter()-started < budget if calls is None else done < calls):
        operator.step()
        done += 1
    return perf_counter()-started


# UB 降幅按来源拆分：initial（初始化）、sa（块采样与块内最好点 pool）、operator（操作程序）。
def decrease_by_source(evaluations):
    groups = dict(initial='initial', initialization_mip='initial', sa='sa', pool='sa', operator='operator')
    totals = dict(initial=0., sa=0., operator=0., other=0.)
    for e in evaluations:
        if e['improved'] and np.isfinite(e['incumbent_before']):
            totals[groups.get(e['source'], 'other')] += e['incumbent_before']-e['incumbent']
    return totals


# 0–deadline 内参考差距（%）对时间的平均；首个可行方案之前按首个方案的差距计。
def auc(trajectory, reference, horizon):
    if not trajectory or reference is None or reference == 0:
        return None
    gaps = [(t, 100*(cost-reference)/abs(reference)) for t, cost in trajectory]
    total, (time, gap) = 0., (0., gaps[0][1])
    for t, g in gaps[1:]:
        t = min(t, horizon)
        total += gap*max(0., t-time)
        time, gap = t, g
    total += gap*max(0., horizon-time)
    return total/horizon


# 一次冷启动运行：SA 只做 SA 轮次；其余方法在每轮 SA 之后让算子运行同样长的时间。
# rounds、burst_calls 只供测试（限定轮数与每段程序调用次数以保证可复现）。
def run_trial(name, arm, seed, seconds, reference, rounds=None, burst_calls=None, *, data, o, progress_callback=None):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm}')
    d, scenarios = data
    trajectory, sa_seconds, initial = [], 0., {}
    started = perf_counter()
    run = solver._Run(o, started, dict(reference), started)

    def progress(whole, event):
        objective = whole.result.objective
        if np.isfinite(objective) and (not trajectory or objective < trajectory[-1][1]):
            trajectory.append((perf_counter()-started, float(objective)))
        if event == 'initialization':
            initial.update(objective=float(objective), u=whole.result.u.astype(int).tolist(),
                           seconds=perf_counter()-started)
        if progress_callback is not None:
            progress_callback(whole, event)

    operator, status, message = None, 'TIME_LIMIT', ''
    try:
        try:
            m, relaxation = solver._initialize(run, d, scenarios, progress)
            whole = run.whole
            if arm != 'SA':
                operator = Operator(whole, seed, PROGRAMS[arm].read_text(encoding='utf-8') if arm in PROGRAMS else None,
                                    arm in SCENARIO_ARMS)
            blocks, cross = partition(m, o.block_hours)
            prices = search_multipliers(m, cross, relaxation.prices[cross])
            run.searches = [search_core.BlockSearch(priced(b, prices), o, j+1, whole) for j, b in enumerate(blocks)]
            k = 0
            while rounds is None or k < rounds:
                k += 1
                phase = perf_counter()
                try:
                    reached = solver._run_round(run, k)
                finally:
                    sa_seconds += perf_counter()-phase
                if reached:
                    status = 'REFERENCE_TARGET_REACHED'
                    break
                if operator is not None:
                    burst(operator, perf_counter()-phase, burst_calls)
                run.require_time()
            else:
                status = 'ROUNDS'
        except TimeoutError:
            status = 'TIME_LIMIT'
        except ProgramError as error:
            status, message = 'PROGRAM_INVALID', str(error)
        except Exception as error:
            status, message = f'{run.stage}_FAILED', f'{type(error).__name__}: {error}'
    finally:
        if operator is not None:
            operator.close()
        solver._shutdown(run)
    solver._finalize(run, d, scenarios)
    if run.result.status == 'VALIDATION_FAILED':
        status, message = run.result.status, run.result.message
    result, whole = run.result, run.whole
    horizon = o.time_limit - solver.DEADLINE_MARGIN_SECONDS
    return dict(instance=name, arm=arm, seed=seed, seconds=seconds, status=status, message=message,
                objective=float(result.objective), reference_cost=result.reference_cost,
                gap_percent=result.reference_gap_percent, feasible=bool((result.validation or {}).get('feasible')),
                auc=auc(trajectory, result.reference_cost, horizon), trajectory=trajectory,
                initial=initial, online_llm_calls=result.llm_calls,
                decrease=decrease_by_source(result.diagnostics.get('evaluations', [])),
                sa_calls=result.diagnostics.get('sa_calls', 0), sa_seconds=sa_seconds,
                operator_calls=operator.calls if operator else 0,
                operator_seconds=operator.seconds if operator else 0.,
                operator_generation_seconds=operator.generation_seconds if operator else 0.,
                operator_outcomes=operator.outcomes if operator else {},
                timings=dict(whole.timings) if whole else {}, final_u=result.u.astype(int).tolist() if result.u is not None else None,
                elapsed_seconds=result.elapsed_seconds)
