"""离线 LLM 程序 L3：沙箱调用、候选检查，以及 SA+Benders 不加（A）与加 L3（L+SA）的一次冷启动运行。"""
# 在线不调用 LLM；L3 是事先写好的冻结程序（programs/L3.py），在隔离子进程中运行。参考最优解只用来计算差距，不注入搜索。
import json
import os
from pathlib import Path
import queue
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
from offline_worker import check_source

ROOT = Path(__file__).resolve().parent
L3 = ROOT/'programs'/'L3.py'
ARMS = ('A', 'L+SA')
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
def context(d, scenarios, plan):
    load = sum(s.probability*np.asarray(s.load, float) for s in scenarios)
    p = sum(s.probability*np.asarray(sp.p, float) for s, sp in zip(scenarios, plan.scenarios))
    return jsonable(dict(n_units=d.n_units, horizon=d.horizon,
        **{k: getattr(d, k) for k in CONTEXT_FIELDS}, load=load,
        reserve_fraction=d.reserve_fraction, u=plan.u, p=p))


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


class Operator:
    def __init__(self, whole, seed, source):
        self.whole, self.seed, self.source = whole, seed, source
        self.program, self.calls, self.empty, self.seconds = None, 0, 0, 0.
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
            if self.program is None:
                self.program = Program(self.source, whole.deadline)
            candidates = self.program.propose(context(whole.d, whole.scenarios, whole.result), seed, whole.deadline)
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
    if not trajectory:
        return None
    gaps = [(t, 100*(cost-reference)/abs(reference)) for t, cost in trajectory]
    total, (time, gap) = 0., (0., gaps[0][1])
    for t, g in gaps[1:]:
        t = min(t, horizon)
        total += gap*max(0., t-time)
        time, gap = t, g
    total += gap*max(0., horizon-time)
    return total/horizon


# 一次冷启动运行：A 只做 SA 轮次；L+SA 在每轮 SA 之后让 L3 运行同样长的时间。
# rounds、burst_calls 只供测试（限定轮数与每段程序调用次数以保证可复现）。
def run_trial(name, arm, seed, seconds, reference, rounds=None, burst_calls=None, *, data, o):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm}')
    d, scenarios = data
    trajectory, sa_seconds = [], 0.
    started = perf_counter()
    run = solver._Run(o, started, dict(reference), started)

    def progress(whole, event):
        objective = whole.result.objective
        if np.isfinite(objective) and (not trajectory or objective < trajectory[-1][1]):
            trajectory.append((perf_counter()-started, float(objective)))

    operator, status, message = None, 'TIME_LIMIT', ''
    try:
        try:
            m, relaxation = solver._initialize(run, d, scenarios, progress)
            whole = run.whole
            if arm == 'L+SA':
                operator = Operator(whole, seed, L3.read_text(encoding='utf-8'))
            blocks, cross = partition(m, o.block_hours)
            prices = search_multipliers(m, cross, relaxation.prices[cross])
            run.searches = [search_core.BlockSearch(priced(b, prices), o, j+1, whole) for j, b in enumerate(blocks)]
            k = 0
            while rounds is None or k < rounds:
                k += 1
                phase = perf_counter()
                if solver._run_round(run, k):
                    status = 'REFERENCE_TARGET_REACHED'
                    break
                sa_seconds += perf_counter()-phase
                if operator is not None:
                    burst(operator, perf_counter()-phase, burst_calls)
                run.require_time()
            else:
                status = 'ROUNDS'
        except TimeoutError:
            status = 'TIME_LIMIT'
        except ProgramError as error:
            status, message = 'PROGRAM_INVALID', str(error)
    finally:
        if operator is not None:
            operator.close()
        solver._shutdown(run)
    solver._finalize(run, d, scenarios)
    result, whole = run.result, run.whole
    horizon = o.time_limit - solver.DEADLINE_MARGIN_SECONDS
    return dict(instance=name, arm=arm, seed=seed, seconds=seconds, status=status, message=message,
                objective=float(result.objective), reference_cost=result.reference_cost,
                gap_percent=result.reference_gap_percent, feasible=bool((result.validation or {}).get('feasible')),
                auc=auc(trajectory, result.reference_cost, horizon), trajectory=trajectory,
                decrease=decrease_by_source(whole.diagnostics['evaluations']),
                sa_calls=whole.diagnostics.get('sa_calls', 0), sa_seconds=sa_seconds,
                operator_calls=operator.calls if operator else 0,
                operator_seconds=operator.seconds if operator else 0.,
                operator_outcomes=operator.outcomes if operator else {},
                timings=dict(whole.timings), final_u=result.u.astype(int).tolist(),
                elapsed_seconds=result.elapsed_seconds)
