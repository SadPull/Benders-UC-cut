"""Fresh C runs, isolated reference snapshots and replayable continuous certificates."""
from time import perf_counter, time
PROCESS_STARTED, PROCESS_EPOCH = perf_counter(), time()

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import gurobipy as gp
import numpy as np
from scipy import sparse

from suc_benders.anneal import anneal, initialize
from suc_benders.benders import coordinate, master
from suc_benders.config import load_options
from suc_benders.model import partition
from suc_benders.optimization import continuous_only, optimize
from suc_benders.paper import STREAMS, decode, record_bound, record_state, sampling_evidence
from suc_benders.qubo import qubo
from suc_benders.solver import BlockSearch, Result, WholeDay
from .azema import GUROBI_VERSION
from .azema.artifacts import append, memory_gate, read_jsonl, save, scored_trace, sha256, snapshot, source_hashes
from .azema.controller import Stopped, require_finished
from .azema.process import run_process
from .azema.protocol import environment
from .paper_audit import audit
from .rts24 import PilotRecorder, case
from .rts24_improved import INPUTS, RTS24, reference_check

BUDGET = 1800.


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def options(directory, seed):
    o = load_options(dict(data_file=str(directory/'inputs/units.json'),
        scenario_file=str(directory/'inputs/stochastic.json'), paper_sa=True,
        bound_mode='relaxation', rts24_improvements=False, sa_enabled=True, llm_enabled=False,
        time_limit=BUDGET, tolerance=.001, block_hours=3, max_outer=20, max_inner=20,
        seed=seed, verbose=True))
    private = {'kaiwu_user_id', 'kaiwu_sdk_code', 'llm_url', 'llm_api_key'}
    return {k: v for k, v in asdict(o).items() if k not in private}


def binding(directory, seed, allow_low_memory):
    return dict(source=source_hashes(), environment=environment(), options=options(directory, seed),
        inputs={name: sha256(directory/'inputs'/name) for name in INPUTS},
        memory_gate_waived=allow_low_memory)


class PaperRecorder(PilotRecorder):
    full_diagnostics = False
    streams = (*PilotRecorder.streams, *STREAMS)

    def __init__(self, directory, budget, started, allow_low_memory=False):
        super().__init__(directory, budget)
        self.started, self.allow_low_memory = started, allow_low_memory

    def optimizer_event(self, event):
        append(self.directory/'optimizer_calls.jsonl', dict(event, elapsed_seconds=perf_counter()-self.started))

    def __call__(self, whole, event):
        if event in ('coordination', 'final'):
            append(self.directory/'block_status.jsonl', dict(event=event,
                elapsed_seconds=perf_counter()-whole.started, blocks=whole.diagnostics['blocks'][-8:]))
        super().__call__(whole, event)


def encode_search(search):
    m, lower = master(search.b, search.cuts, search.whole.remaining())
    record_bound(search, m)
    search.lower = max(search.lower, lower)
    m.lb[-1] = max(m.lb[-1], search.lower-search.b.constant)
    q = qubo(m, search.o, search.upper-search.b.constant)
    if not np.isfinite(q.Q.data).all():
        raise ValueError('Nonfinite repriced QUBO')
    return m, q


def save_snapshot(search, folder, m, q):
    folder.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(folder/'master.npz', m.A)
    sparse.save_npz(folder/'qubo.npz', q.Q)
    np.savez_compressed(folder/'arrays.npz', rhs=m.rhs, eq=m.eq, obj=m.obj, q=m.q,
        lb=m.lb, ub=m.ub, binary=m.binary, fixed=q.fixed, free=q.free,
        theta_ids=q.theta_ids, theta_weights=q.theta_weights)
    seed = int(np.random.SeedSequence([search.o.seed, search.outer, 0, search.index]).generate_state(1)[0] % (2**31-1))
    proposals, sampling = anneal(q, search.o, search.whole.remaining(), seed)
    sampling_evidence(search, m, q, sampling)
    sampling.update(seed=seed, raw_candidates=len(proposals),
                    unique_uvw=len({x.tobytes() for x in proposals}))
    save(folder/'samples.json', sampling)
    save(folder/'model.json', dict(context=search.context, outer=search.outer, block=search.index,
        nx=m.nx, constant=search.b.constant, offset=q.offset, theta_lower=q.theta_lower,
        bits=q.Q.shape[0], native_lower=search.lower, native_upper=search.upper,
        sha256={name: sha256(folder/name) for name in ('master.npz', 'qubo.npz', 'arrays.npz', 'samples.json')}))


def snapshot_reference(folder, remaining):
    meta = read(folder/'model.json')
    for name, digest in meta['sha256'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Reference snapshot hash changed')
    arrays = dict(np.load(folder/'arrays.npz', allow_pickle=False))
    from types import SimpleNamespace as NS
    m = NS(A=sparse.load_npz(folder/'master.npz'), **{k: arrays[k] for k in
           ('rhs', 'eq', 'obj', 'q', 'lb', 'ub', 'binary')})
    started = perf_counter()
    native = optimize(m, min(30., remaining()), integral=True)
    evidence = dict(native=dict(status=native.status, seconds=perf_counter()-started,
        objective=native.objval+meta['constant'], lower=native.lower+meta['constant'],
        certified=native.status == 'OPTIMAL'))
    save(folder/'reference.json', evidence)
    if remaining() <= 0:
        raise TimeoutError('Isolated reference budget exhausted')
    Q = sparse.load_npz(folder/'qubo.npz'); scale = max(1., float(abs(Q).max()))
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        with gp.Model(env=env) as g:
            for key, value in dict(Threads=1, TimeLimit=min(30., remaining()), MIPGap=0,
                    MIPGapAbs=1e-8/scale, NonConvex=2, Seed=7, FeasibilityTol=1e-8).items():
                g.setParam(key, value)
            x = g.addMVar(Q.shape[0], vtype=gp.GRB.BINARY)
            g.setObjective(x @ (Q/scale) @ x)
            started = perf_counter(); g.optimize()
            result = dict(status=int(g.Status), seconds=perf_counter()-started,
                          certified=g.Status == gp.GRB.OPTIMAL, integer_variables=Q.shape[0])
            try:
                result['lower_energy'] = float(g.ObjBound)*scale+meta['offset']
            except AttributeError:
                pass
            if g.SolCount:
                bits = np.rint(x.X).astype(np.uint8)
                result['bits'] = ''.join(map(str, bits))
                result['energy'] = float(bits @ Q @ bits+meta['offset'])
                q = NS(Q=Q, fixed=arrays['fixed'], free=arrays['free'], theta_ids=arrays['theta_ids'],
                       theta_weights=arrays['theta_weights'], theta_lower=meta['theta_lower'])
                point = decode(q, result['bits']); residual = m.A@point-m.rhs
                residual[m.eq] = abs(residual[m.eq])
                result['native_residual'] = float(residual.max(initial=0))
                result['native_master_feasible'] = result['native_residual'] <= 1e-7
                result['decoded_objective'] = float(point[-1]+meta['constant'])
            evidence['qubo'] = result
    save(folder/'reference.json', evidence)
    return evidence


def numerical_preflight(job, started):
    folder = Path(job['directory']); o = load_options(job['options'])
    recorder = PaperRecorder(folder/'search', job['time_limit'], started, job['allow_low_memory'])
    m = case(Path(job['inputs'])); full = partition(m, 24)[0][0]; full.audit_cuts = True
    whole = WholeDay(m, full, o, started+job['time_limit']-5, Result(), started, recorder)
    snapshots, reencodings = [], []
    with continuous_only(recorder.optimizer_event):
        initialize(o.kaiwu_user_id, o.kaiwu_sdk_code)
        initial = optimize(m, whole.remaining())
        if initial.status != 'OPTIMAL' or not np.isfinite(initial.lower):
            raise ValueError('Preflight initialization failed')
        whole.result.lower_bound = initial.lower; recorder.relaxation(m, initial, perf_counter()-started)
        for raw in (np.floor(initial.x[m.ix.u]+.5), np.ones((32, 24))):
            record = {}; whole.accept(whole.prepare(raw, 'initial', record), 'initial', record)
        if whole.best is None:
            raise ValueError('No initial feasible schedule')
        prices, previous, planes = initial.prices, [[] for _ in range(8)], [[] for _ in range(8)]
        for outer in (1, 2, 3):
            record_state(whole, prices, outer)
            blocks, cross = partition(m, 3, prices)
            searches = [BlockSearch(b, o, outer, j+1, whole, planes[j]) for j, b in enumerate(blocks)]
            for j, search in enumerate(searches):
                for seed in [whole.best.point[search.b.cols[search.b.binary]], *previous[j]]:
                    search.evaluate(seed, 'initial')
                native, q = encode_search(search)
                if outer in (1, 3):
                    target = folder/'snapshots'/f'state-{outer}-block-{j+1}'
                    save_snapshot(search, target, native, q); snapshots.append(target)
                if outer > 1:
                    reencodings.append(dict(after_coordination=outer-1, block=j+1, bits=q.Q.shape[0],
                                            max_coefficient=float(abs(q.Q).max())))
            if outer == 3:
                break
            for inner in (1, 2, 3):
                for search in searches:
                    search.step([whole.best.point[search.b.cols[search.b.binary]]], inner)
                whole.diagnostics['complete_rounds'] += 1
            values, upper = coordinate(planes, m.eq[cross], whole.result.objective, whole.remaining())
            previous = [s.seeds for s in searches]
            prices = np.zeros(len(m.rhs)); prices[cross] = values
            whole.diagnostics['coordination_count'] += 1
            whole.emit('coordination')
        whole.result.status = 'PREFLIGHT_COMPLETE'; whole.emit('final')
    if len(reencodings) != 16 or len(snapshots) != 16:
        raise ValueError('Incomplete eight-block repricing or reference snapshots')
    save(folder/'sa_check.json', dict(passed=True, coordination_count=2, reencodings=reencodings,
        raw_search_sa_calls=whole.diagnostics['sa_calls'], isolated_snapshot_sa_calls=16))
    # These integer optimizations start only after the preflight search is finished.
    references = []
    for path in snapshots:
        references.append(dict(snapshot=path.name, **snapshot_reference(path, whole.remaining)))
        save(folder/'references.json', references)
    with continuous_only(recorder.optimizer_event):
        audit(folder/'search', job['inputs'], job['time_limit'], folder/'audit', whole.remaining)
        reference_check(dict(inputs=job['inputs'], directory=str(folder/'reference'), time_limit=whole.remaining()))
    save(folder/'check.json', dict(passed=True, snapshots=16, reencodings=16, coordination_count=2,
        completed_seconds=perf_counter()-started, reference=str(folder/'reference/check.json')))


def worker_main(job):
    folder = Path(job['directory']); gate = memory_gate(job.get('estimated_peak_bytes', 0))
    gate['waived_by_user'] = job['allow_low_memory']; save(folder/'resource_gate.json', gate)
    if not gate['passed'] and not job['allow_low_memory']:
        save(folder/'state.json', dict(status='BLOCKED_MEMORY', numerical_runs_started=0)); return 3
    if '.'.join(map(str, gp.gurobi.version())) != GUROBI_VERSION:
        raise ValueError('Use the isolated Gurobi 12.0.2 environment')
    started = PROCESS_STARTED-max(0., PROCESS_EPOCH-job['launch_epoch_seconds'])
    if job['action'] == 'preflight':
        numerical_preflight(job, started)
    elif job['action'] == 'run':
        from run_suc import run_suc
        result = run_suc(job['options'], progress=PaperRecorder(folder, BUDGET, started, job['allow_low_memory']), started_at=started)
        save(folder/'result.json', {k: getattr(result, k) for k in
             ('status', 'message', 'objective', 'lower_bound', 'relative_gap', 'elapsed_seconds')})
        if result.status not in ('CONVERGED', 'TIME_LIMIT', 'MAX_ITERATIONS'):
            raise RuntimeError(result.status+': '+result.message)
    elif job['action'] == 'audit':
        with continuous_only(lambda event: append(folder/'optimizer_calls.jsonl', event)):
            audit(job['run_directory'], job['inputs'], BUDGET, folder,
                  lambda: started+job['time_limit']-perf_counter())
    else:
        raise ValueError('Unknown paper-C worker action')
    return 0


def launch(action, folder, seconds, **values):
    job = dict(action=action, directory=str(folder), time_limit=seconds, **values)
    save(folder/'job.json', job)
    job['launch_epoch_seconds'] = time(); save(folder/'job.json', job)
    state = run_process([sys.executable, '-B', '-m', 'benchmarks.rts24_paper_c', str(folder/'job.json')],
        folder, seconds, enforce_memory_gate=not values['allow_low_memory'],
        estimated_peak_bytes=values.get('estimated_peak_bytes', 0))
    if action != 'run' or state['status'] not in ('FINISHED', 'TIME_LIMIT'):
        require_finished(state, action)
    return state


def write_report(directory, seed):
    folder = directory/'C'/f'seed-{seed}'
    rows = scored_trace(folder, BUDGET)
    if not rows:
        save(directory/'summary.json', dict(status='NO_SCORED_RUN', state=read(directory/'state.json')))
        return
    final = rows[-1]; state = read(folder/final['checkpoint'])
    audited = read(folder/'audit/audit.json', {})
    valid = audited.get('passed', False) and audited['final'] == final
    preflight = read(directory/'preflight.json')
    reference = read(Path(preflight['numerical_directory'])/'reference/check.json')
    sampling = read_jsonl(folder/'sampling.jsonl')[:state['offsets']['sampling']]
    candidates = read_jsonl(folder/'paper_candidates.jsonl')[:state['offsets']['paper_candidates']]
    evaluations = read_jsonl(folder/'evaluations.jsonl')[:state['offsets']['evaluations']]
    raw = sum(r.get('raw_candidates', 0) for r in sampling)
    repair_distances = [r['hamming_u'] for r in candidates if 'hamming_u' in r]
    upper_changes = {}
    for row in evaluations:
        if row.get('incumbent_before') is not None:
            upper_changes[row['source']] = upper_changes.get(row['source'], 0.)+max(0., row['incumbent_before']-row['incumbent'])
    snapshots = {str(t): next((r for r in reversed(rows) if r['elapsed_seconds'] <= t), None)
                 for t in (60, 300, 600, 1200, 1800)}
    met = lambda r: r['objective']-r['lower_bound'] <= 1e-5+.001*max(1., abs(r['objective']))
    result = dict(arm='C', seed=seed, budget_seconds=BUDGET, final=final, audit_passed=valid,
        target_met=bool(valid and met(final)), target_seconds=next((r['elapsed_seconds'] for r in rows if met(r)), None) if valid else None,
        stop_reason=read(folder/'result.json', {}).get('status', read(folder/'state.json', {}).get('status')),
        snapshots=snapshots, reference_cost=reference['objective'], reference_lower=reference['lower_bound'],
        reference_cost_gap=(final['objective']-reference['objective'])/abs(reference['objective']),
        timings=state['timings'], process=read(folder/'state.json'),
        complete_rounds=state['complete_rounds'], coordination_count=state['coordination_count'],
        initialization_cost=min(r['objective'] for r in evaluations if r['source'] == 'initial' and r['feasible']),
        upper_decrease_by_source=upper_changes, memory_gate_waived=read(directory/'protocol.json')['memory_gate_waived'],
        gurobi_integer_calls=audited.get('gurobi_integer_calls'), gurobi_continuous_calls=audited.get('gurobi_continuous_calls'),
        local_cuts=audited.get('local_cuts'), lp_proofs=audited.get('lp_proofs'),
        sa=dict(calls=final['sa_calls'], raw_candidates=raw,
            raw_discrete_feasible=sum(r.get('raw_discrete_feasible', 0) for r in sampling),
            native_master_feasible=sum(sum(r.get('native_master_feasible', [])) for r in sampling),
            within_batch_duplicate_rate=1-sum(r.get('unique_uvw', 0) for r in sampling)/raw if raw else None,
            qubo_bits_min=min((r['bits'] for r in sampling), default=None),
            qubo_bits_mean=np.mean([r['bits'] for r in sampling]) if sampling else None,
            qubo_bits_max=max((r['bits'] for r in sampling), default=None),
            mean_repair_u=float(np.mean(repair_distances)) if repair_distances else None,
            repair_outcomes=dict(Counter(r.get('outcome') for r in candidates))))
    if valid:
        result['cost_breakdown'] = read(folder/'audit'/audited['final_solution'])
    save(directory/'summary.json', result)
    lines = ['# RTS-24 论文分工 C 组实验', '',
        f'状态：{result["stop_reason"]}；正确性审计：{valid}；自有认证 gap ≤0.1%：{result["target_met"]}。', '',
        '整数主问题使用 Kaiwu SA。Gurobi 仅用于连续初始化、调度、主问题 LP 松弛和协调。',
        f'本次用户明确取消 4 GiB 启动门槛：{result["memory_gate_waived"]}；资源监控和 1800 秒看门狗仍启用。', '',
        '| 秒 | UB | 认证 LB | 认证 gap | 参考成本差 |', '|---:|---:|---:|---:|---:|']
    for t, row in snapshots.items():
        if row:
            difference=(row['objective']-reference['objective'])/abs(reference['objective'])
            lines.append(f'| {t} | {row["objective"]:.6f} | {row["lower_bound"]:.6f} | {100*row["relative_gap"]:.6f}% | {100*difference:.6f}% |')
        else:
            lines.append(f'| {t} | 尚无预算内方案 | — | — | — |')
    lines += ['', f'最终 UB {final["objective"]:.6f}，LB {final["lower_bound"]:.6f}，认证 gap {100*final["relative_gap"]:.6f}%。',
        f'Gurobi 参考成本 {reference["objective"]:.6f}、参考下界 {reference["lower_bound"]:.6f}；参考成本差 {100*result["reference_cost_gap"]:.6f}%。',
        f'正式整数 Gurobi 调用 {result["gurobi_integer_calls"]} 次；连续调用 {result["gurobi_continuous_calls"]} 次；LLM 调用 {final["llm_calls"]} 次。',
        f'完整 SA 轮次 {result["complete_rounds"]}，协调 {result["coordination_count"]} 次，有效局部割 {result["local_cuts"]} 条。',
        '', '## 来源与统计', '', f'初始化成本：{result["initialization_cost"]:.6f}。',
        f'后续降本分来源：{json.dumps(upper_changes, ensure_ascii=False)}。',
        f'SA 统计：{json.dumps(result["sa"], ensure_ascii=False)}。',
        f'分项耗时（秒）：{json.dumps(result["timings"], ensure_ascii=False)}。']
    if valid:
        value=result['cost_breakdown']
        lines += ['', '## 成本复核', '', '| 项目 | C | Gurobi 参考 |', '|---|---:|---:|',
            f'| 固定费用 | {value["fixed_cost"]:.6f} | {reference["fixed_cost"]:.6f} |',
            f'| 期望燃料费用 | {value["objective"]-value["fixed_cost"]:.6f} | {reference["objective"]-reference["fixed_cost"]:.6f} |']
        for a,b in zip(value['scenarios'], reference['scenarios']):
            lines.append(f'| {a["name"]} 场景燃料 | {a["objective"]:.6f} | {b["objective"]:.6f} |')
    lines += ['', '认证 gap=(UB−LB)/|UB|；参考成本差=(UB−参考成本)/|参考成本|。',
        'SA 能量、离线整数参考及参考下界均不进入正式下界。预检不构成正式成绩；未达标不自动延长预算。',
        '本实验保留 RTS-24 随机场景和二次费用，属于论文框架的 SA 实现，不是原文算例或 CIM 性能复现。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def execute(directory, phase, seed=7, allow_low_memory=False):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    if phase == 'report':
        write_report(directory, seed); return 0
    if not (directory/'protocol.json').exists():
        for name in INPUTS:
            target=directory/'inputs'/name; target.parent.mkdir(exist_ok=True)
            shutil.copy2(RTS24/'inputs'/name, target)
        save(directory/'protocol.json', dict(arm='C', seed=seed, time_limit=BUDGET, tolerance=.001,
            master_backend='kaiwu_sa', classical_backend='continuous_gurobi',
            offline_reference_feedback=False, memory_gate_waived=allow_low_memory,
            memory_policy_source='User explicitly waived the 4 GiB gate for this experiment' if allow_low_memory else '4 GiB gate'))
    if read(directory/'protocol.json')['memory_gate_waived'] != allow_low_memory:
        raise ValueError('Use the recorded explicit memory policy')
    folder=directory/'C'/f'seed-{seed}'
    try:
        save(directory/'state.json', dict(status='RUNNING', phase=phase))
        if phase in ('all', 'preflight'):
            if (folder/'trace.jsonl').exists():
                raise ValueError('Formal evidence exists; do not replace its preflight')
            attempt=directory/'preflight'/f'attempt-{len(list((directory/"preflight").glob("attempt-*")))+1:03d}'
            snapshot(attempt)
            evidence=dict(status='RUNNING', binding=binding(directory, seed, allow_low_memory), numerical_directory=str(attempt/'numerical'))
            save(directory/'preflight.json', evidence)
            xml=attempt/'regression/pytest.xml'
            state=run_process([sys.executable, '-B', '-m', 'pytest', '-q', '-m', 'not integration', '-p', 'no:cacheprovider',
                f'--junitxml={xml}'], attempt/'regression', BUDGET, enforce_memory_gate=not allow_low_memory,
                env={'SUC_EXPERIMENT_ALLOW_LOW_MEMORY': '1' if allow_low_memory else '0'})
            require_finished(state, 'regression')
            if any(int(s.get(k, 0)) for s in ET.parse(xml).getroot().iter('testsuite') for k in ('failures', 'errors', 'skipped')):
                raise ValueError('Core regression failed or skipped required checks')
            state=launch('preflight', attempt/'numerical', BUDGET, inputs=str(directory/'inputs'),
                options=options(directory, seed), allow_low_memory=allow_low_memory)
            if not read(attempt/'numerical/check.json')['passed']:
                raise ValueError('Numerical preflight did not pass')
            evidence.update(status='PASSED', estimated_peak_bytes=state['peak_rss_bytes']); save(directory/'preflight.json', evidence)
        evidence=read(directory/'preflight.json', {})
        if evidence.get('status') != 'PASSED' or evidence.get('binding') != binding(directory, seed, allow_low_memory):
            raise ValueError('Preflight missing, incomplete or stale')
        if phase in ('all', 'run'):
            if read(folder/'state.json', {}).get('numerical_runs_started'):
                raise ValueError('Formal run already started; refusing an implicit rerun')
            save(folder/'manifest.json', dict(binding=evidence['binding'], fresh_initialization=True, preflight_feedback=False))
            launch('run', folder, BUDGET, options=options(directory, seed), allow_low_memory=allow_low_memory,
                   estimated_peak_bytes=evidence['estimated_peak_bytes'])
        if phase in ('all', 'run', 'audit') and (folder/'trace.jsonl').exists():
            launch('audit', folder/'audit', 900, inputs=str(directory/'inputs'), run_directory=str(folder),
                   allow_low_memory=allow_low_memory)
            write_report(directory, seed)
        save(directory/'state.json', dict(status='FINISHED', phase=phase))
        return 0
    except Stopped as error:
        save(directory/'state.json', dict(status=error.state['status'], stage=error.stage, detail=error.state)); return 3
    except Exception as error:
        save(directory/'state.json', dict(status='FAILED', error=type(error).__name__, message=str(error))); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('job', type=Path)
    raise SystemExit(worker_main(read(parser.parse_args().job)))
