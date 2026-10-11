"""Independent C/D runs with strict wall budgets, fresh starts and audited evidence."""
from time import perf_counter, time
PROCESS_STARTED, PROCESS_EPOCH = perf_counter(), time()
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import gurobipy as gp
import numpy as np

from suc_benders.config import load_options
from suc_benders.model import fingerprint, partition
from suc_benders.scenario_bounds import ScenarioBounds
from suc_benders.validation import pack
from .rts24 import PilotRecorder, case, fixed_check, quadratic_dual
from .azema import GUROBI_VERSION, ROOT
from .azema.artifacts import append, memory_gate, read_jsonl, save, scored_trace, sha256, snapshot, source_hashes
from .azema.audit import verify_cut, verify_trace
from .azema.controller import Stopped, require_finished, worker
from .azema.process import run_process
from .azema.protocol import environment

RTS24 = ROOT/'data/rts24'
BUDGET = 1800.
INPUTS = ('units.json', 'stochastic.json', 'provenance.json')


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def options(directory, arm, seed, run_directory):
    if arm not in ('C', 'D') or seed not in (7, 11, 19):
        raise ValueError('Only C/D and seeds 7,11,19 are approved')
    o = load_options(dict(data_file=str(directory/'inputs/units.json'),
        scenario_file=str(directory/'inputs/stochastic.json'), bound_mode='integer_benders',
        rts24_improvements=True, time_limit=BUDGET, tolerance=.001, seed=seed,
        block_hours=3, max_outer=20, max_inner=3, bound_slice_seconds=30,
        sa_enabled=True, llm_enabled=arm == 'D', llm_model='gpt-6-astra',
        llm_agent_directory=str(run_directory/'agent_bridge'), llm_timeout=600,
        llm_patience=3, llm_max_calls=12, verbose=True))
    private = {'kaiwu_user_id', 'kaiwu_sdk_code', 'llm_url', 'llm_api_key'}
    return {k: v for k, v in asdict(o).items() if k not in private}


def binding(directory):
    return dict(source=source_hashes(), environment=environment(),
        inputs={name: sha256(directory/'inputs'/name) for name in INPUTS},
        options={a: options(directory, a, 7, directory/a/'seed-7') for a in ('C', 'D')})


class ImprovementRecorder(PilotRecorder):
    # The streamed evidence is authoritative; avoid repacking it at the wall deadline.
    full_diagnostics = False
    streams = (*PilotRecorder.streams, 'scenario_certificate_cuts', 'native_bounds')

    def __call__(self, whole, event):
        # Final revisions are appended explicitly, rather than relying on first-written mutable rows.
        records = {'probe_complete': whole.diagnostics.get('bound_probes', []),
                   'certification': whole.result.certification_history,
                   'round': whole.diagnostics.get('rounds', []),
                   'repricing': whole.diagnostics.get('repricing_checks', []),
                   'llm_response': whole.result.llm_history}
        if event in records and records[event]:
            append(self.directory/'completed_events.jsonl', dict(event=event,
                elapsed_seconds=perf_counter()-whole.started, value=records[event][-1]))
        super().__call__(whole, event)


def preflight_whole(job):
    from .azema.worker import read_common
    from suc_benders.solver import WholeDay, Result
    o = load_options(job['options']); o.time_limit = job['time_limit']
    m = case(Path(job['inputs'])); full = partition(m, m.d.horizon)[0][0]; full.audit_cuts = True
    started = perf_counter()
    recorder = ImprovementRecorder(job['directory'], o.time_limit)
    whole = WholeDay(m, full, o, started+o.time_limit-5, Result(), started, recorder)
    common = read_common(job['common_directory'])
    if common['model_fingerprint'] != fingerprint(m):
        raise ValueError('Preflight model mismatch')
    whole.result.lower_bound = common['lower_bound']
    recorder.relaxation(m, NS(lower=common['lower_bound'], prices=common['relaxation_prices']), 0.)
    whole.accept(pack(m.d, common['u']), 'initial', {'proposal_id': 'preflight:initial'})
    return whole, common['relaxation_prices']


def real_sa(job):
    from suc_benders.anneal import initialize
    from suc_benders.benders import coordinate
    from suc_benders.solver import BlockSearch
    whole, prices = preflight_whole(job)
    try:
        initialize(whole.o.kaiwu_user_id, whole.o.kaiwu_sdk_code)
        planes = [[] for _ in range(8)]
        for outer in (1, 2):
            blocks, cross = partition(whole.m, 3, prices)
            assert len(blocks) == 8
            searches = [BlockSearch(b, whole.o, outer, j+1, whole, planes[j]) for j, b in enumerate(blocks)]
            for inner in (1, 2, 3):
                for search in searches:
                    search.step([whole.best.point[search.b.cols[search.b.binary]]], inner)
            values, upper = coordinate(planes, whole.m.eq[cross], whole.result.objective, whole.remaining())
            prices = np.zeros(len(whole.m.rhs)); prices[cross] = values
            whole.diagnostics['coordination_count'] += 1
            whole.check_repricing(prices, outer)
        whole.emit('final')
        checks = whole.diagnostics['repricing_checks']
        if len(checks) != 2 or any(len(r['blocks']) != 8 for r in checks):
            raise ValueError('Two full repricing checks did not finish')
        save(Path(job['directory'])/'check.json', dict(passed=True, coordination_count=2,
            real_sa_calls=whole.diagnostics['sa_calls'], checks=checks,
            dense_working_bytes=max(b['dense_working_bytes'] for r in checks for b in r['blocks'])))
    finally:
        whole.certifier.close()


def real_llm(job):
    whole, _ = preflight_whole(job)
    try:
        whole.certifier.step()
        if whole.certifier.last_master is None:
            raise ValueError('No lower-channel context')
        for channel in ('upper', 'lower'):
            whole.cache.clear()
            whole.advice.begin([channel], validation=True)
            whole.advice.run_batch([])
        for channel in ('upper', 'lower'):
            if not any(r['channel'] == channel and r['status'] == 'OK' and r.get('numerical_evaluation_completed')
                       and r.get('response_model') == 'gpt-6-astra' and r.get('reasoning_effort') == 'max'
                       for r in whole.result.llm_history):
                raise ValueError('Missing actual numerically evaluated '+channel+' proposal')
        whole.emit('final')
        save(Path(job['directory'])/'check.json', dict(passed=True, formal_calls_counted=0,
             receipts=whole.result.llm_history))
    finally:
        whole.advice.close(); whole.certifier.close()


def audit_run(job):
    run, folder = Path(job['run_directory']), Path(job['directory'])
    m = case(Path(job['inputs'])); full = partition(m, m.d.horizon)[0][0]
    from suc_benders.hourly_bounds import HourlyBounds
    hourly, scenarios = HourlyBounds(full), ScenarioBounds(full, m)
    initial = read(run/'initial_bound.json')
    if initial['model_fingerprint'] != fingerprint(m) or sha256(run/'initial_prices.npz') != initial['arrays_sha256']:
        raise ValueError('Initial certificate mismatch')
    lower = quadratic_dual(m, np.load(run/'initial_prices.npz', allow_pickle=False)['prices'])
    if initial['lower_bound'] > lower+1e-5:
        raise ValueError('Invalid initial quadratic bound')
    rows = scored_trace(run, job.get('budget', BUDGET)); verify_trace(rows)
    if not rows:
        raise ValueError('No scored checkpoint')
    final_state = read(run/rows[-1]['checkpoint']); offsets = final_state['offsets']
    streams = {name: read_jsonl(run/(name+'.jsonl'))[:offsets.get(name, 0)] for name in
               ('certificate_cuts', 'hourly_certificate_cuts', 'scenario_certificate_cuts', 'native_bounds')}
    for cut in streams['certificate_cuts']:
        verify_cut(full, cut)
    for cut in streams['hourly_certificate_cuts']:
        verify_cut(hourly.parts[cut['hour']-1].b, dict(cut, optimality=True))
    for cut in streams['scenario_certificate_cuts']:
        verify_cut(scenarios.parts[cut['scenario']].b, cut)
    known = initial['lower_bound']; native = streams['native_bounds']
    for bound in native:
        if (bound['floor_bound'] > known+1e-5 or bound['constant'] != full.constant
            or not np.isfinite(bound['master_bound'])):
            raise ValueError('Invalid native bound provenance')
        for key, stream in (('cuts', 'certificate_cuts'), ('hourly_cuts', 'hourly_certificate_cuts'),
                            ('scenario_cuts', 'scenario_certificate_cuts')):
            used = streams[stream][:bound[key]]
            if len(used) != bound[key] or any(c['elapsed_seconds'] > bound['elapsed_seconds'] for c in used):
                raise ValueError('Native bound refers to future or missing cuts')
        known = max(known, bound['master_bound'])
    schedules = {}
    for row in rows:
        state = read(run/row['checkpoint'])
        if state['model_fingerprint'] != fingerprint(m):
            raise ValueError('Scored model mismatch')
        supported = max([initial['lower_bound'], *(b['master_bound'] for b in native[:state['offsets']['native_bounds']])])
        if row['lower_bound'] > supported+1e-5:
            raise ValueError('Scored bound exceeds its evidence')
        schedules[np.asarray(state['u'], np.uint8).tobytes()] = (state['u'], row['objective'])
    checked = []
    started = perf_counter()
    for i, (u, cost) in enumerate(schedules.values()):
        value = fixed_check(m, u, cost, job['time_limit']-(perf_counter()-started))
        if not value['validation']['feasible']:
            raise ValueError('Final schedule is infeasible')
        path = folder/f'incumbent-{i+1:03d}.json'
        save(path, dict(u=u, **value))
        checked.append(dict(artifact=path.name, objective=value['objective'], validation=value['validation']))
    save(folder/'audit.json', dict(passed=True, final=rows[-1], model_fingerprint=fingerprint(case(Path(job['inputs']))),
        initial_dual_value=lower, incumbents=checked, final_solution=checked[-1]['artifact'],
        cuts={k: len(v) for k, v in streams.items()}))


def reference_check(job):
    m = case(Path(job['inputs']))
    manifest = read(RTS24/'reference/manifest.json'); reference = read(RTS24/'reference/result.json')
    if manifest['model_fingerprint'] != fingerprint(m) or manifest['input_sha256'] != {
        name: sha256(Path(job['inputs'])/name) for name in INPUTS}:
        raise ValueError('Reference inputs changed')
    checked = fixed_check(m, reference['u'], reference['checked']['objective'], job['time_limit'])
    save(Path(job['directory'])/'check.json', dict(passed=True, source=str(RTS24/'reference/result.json'),
        source_sha256=sha256(RTS24/'reference/result.json'), lower_bound=reference['lower_bound'],
        solver_gap=reference['relative_gap'], original_solve_seconds=reference['solve_wall_seconds'], **checked))


def worker_main(job):
    folder = Path(job['directory']); gate = memory_gate(job.get('estimated_peak_bytes', 0))
    save(folder/'resource_gate.json', gate)
    if not gate['passed']:
        save(folder/'state.json', dict(status='BLOCKED_MEMORY', numerical_runs_started=0))
        return 3
    if '.'.join(map(str, gp.gurobi.version())) != GUROBI_VERSION:
        raise ValueError('Use Gurobi 12.0.2 environment')
    action = job['action']
    if action == 'run':
        from run_suc import run_suc
        started = PROCESS_STARTED-max(0., PROCESS_EPOCH-job['launch_epoch_seconds'])
        result = run_suc(job['options'], progress=ImprovementRecorder(folder, BUDGET), started_at=started)
        save(folder/'result.json', {key: getattr(result, key) for key in (
            'status', 'message', 'objective', 'lower_bound', 'relative_gap', 'elapsed_seconds')})
        if result.status not in ('CONVERGED', 'TIME_LIMIT'):
            raise RuntimeError(result.status+': '+result.message)
    else:
        {'sa': real_sa, 'llm': real_llm, 'audit': audit_run, 'reference': reference_check}[action](job)
    return 0


def launch(action, folder, seconds, **values):
    job = dict(action=action, directory=str(folder), time_limit=seconds, **values)
    save(folder/'job.json', job)
    if action == 'run':
        job['launch_epoch_seconds'] = time(); save(folder/'job.json', job)
    state = run_process([sys.executable, '-B', '-m', 'benchmarks.rts24_improved', str(folder/'job.json')],
                        folder, seconds, estimated_peak_bytes=values.get('estimated_peak_bytes', 0))
    if action != 'run' or state['status'] not in ('FINISHED', 'TIME_LIMIT'):
        require_finished(state, action)
    return state


def preflight(directory):
    folder = directory/'preflight'
    evidence = dict(status='RUNNING', binding=binding(directory), passed_checks=[], estimated_peak_bytes=0)
    save(directory/'preflight.json', evidence)
    def completed(name, state):
        evidence['passed_checks'].append(name)
        evidence['estimated_peak_bytes'] = max(evidence['estimated_peak_bytes'], state['peak_rss_bytes'])
        save(directory/'preflight.json', evidence)
    xml = folder/'regression/pytest.xml'
    state = run_process([sys.executable, '-B', '-m', 'pytest', '-q', '-m', 'not integration',
        '-p', 'no:cacheprovider', f'--junitxml={xml}'], folder/'regression', BUDGET)
    require_finished(state, 'core_regression')
    if any(int(s.get(k, 0)) for s in ET.parse(xml).getroot().iter('testsuite') for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Core regression failed or skipped')
    completed('core_regression', state)
    common = folder/'common'
    state = worker('common', common, 600, data_file=str(directory/'inputs/units.json'),
                   scenario_file=str(directory/'inputs/stochastic.json'))
    completed('independent_preflight_initialization', state)
    state = launch('reference', folder/'reference', 120, inputs=str(directory/'inputs'))
    completed('reference_validation', state)
    state = launch('sa', folder/'sa', 600, inputs=str(directory/'inputs'), common_directory=str(common),
                   options=options(directory, 'C', 7, folder/'sa'))
    if not read(folder/'sa/check.json')['passed']:
        raise ValueError('SA check failed')
    evidence['estimated_peak_bytes'] = max(evidence['estimated_peak_bytes'], read(folder/'sa/check.json')['dense_working_bytes'])
    completed('two_coordinations_and_sixteen_reencodings', state)
    state = launch('llm', folder/'llm', BUDGET, inputs=str(directory/'inputs'), common_directory=str(common),
                   options=options(directory, 'D', 7, folder/'llm'))
    if not read(folder/'llm/check.json')['passed']:
        raise ValueError('LLM check failed')
    completed('real_upper_and_lower_subagents', state)
    state = launch('audit', folder/'llm_audit', 600, run_directory=str(folder/'llm'),
                   inputs=str(directory/'inputs'), budget=BUDGET)
    completed('preflight_cut_and_bound_audit', state)
    evidence['status'] = 'PASSED'; save(directory/'preflight.json', evidence)


def summarize_run(directory, arm, seed):
    folder = directory/arm/f'seed-{seed}'
    rows = scored_trace(folder, BUDGET)
    if not rows:
        return None
    verify_trace(rows); final = rows[-1]; checkpoint = read(folder/final['checkpoint'])
    audit = read(folder/'audit/audit.json', {})
    valid = audit.get('passed', False) and audit['final'] == final
    reference = read(directory/'preflight/reference/check.json', {})
    events = read_jsonl(folder/'completed_events.jsonl')
    completed = [r for r in events if r['elapsed_seconds'] <= final['elapsed_seconds']]
    sampling = read_jsonl(folder/'sampling.jsonl')[:checkpoint['offsets']['sampling']]
    evaluations = read_jsonl(folder/'evaluations.jsonl')[:checkpoint['offsets']['evaluations']]
    proposals = read_jsonl(folder/'proposals.jsonl')[:checkpoint['offsets']['proposals']]
    receipts = [r for r in read(folder/'llm_receipts.json', []) if r['started_seconds'] <= final['elapsed_seconds']]
    calls = [{k: r.get(k) for k in ('call_id', 'channel', 'status', 'agent_id', 'response_model', 'reasoning_effort',
        'request_sha256', 'reply_sha256', 'seconds', 'consumed_seconds', 'format_valid', 'accepted',
        'new_global_cuts', 'direct_incumbent_decrease', 'numerical_evaluation_completed')} for r in receipts]
    decreases = {}
    for e in evaluations:
        if e.get('incumbent_before') is not None:
            decreases[e['source']] = decreases.get(e['source'], 0.)+max(0., e['incumbent_before']-e['incumbent'])
    cost = reference.get('objective')
    target = lambda r: r['objective']-r['lower_bound'] <= 1e-5+.001*max(1., abs(r['objective']))
    result = dict(arm=arm, seed=seed, budget_seconds=BUDGET, final=final, audit_passed=valid,
        target_met=bool(valid and target(final)), process=read(folder/'state.json', {}),
        stop_reason=read(folder/'result.json', {}).get('status', read(folder/'state.json', {}).get('status')),
        snapshots={str(t): next((r for r in reversed(rows) if r['elapsed_seconds'] <= t), None)
                   for t in (60, 300, 600, 1200, 1800)},
        target_seconds=next((r['elapsed_seconds'] for r in rows if target(r)), None) if valid else None,
        timings=checkpoint['timings'], complete_rounds=checkpoint['complete_rounds'],
        coordination_count=checkpoint['coordination_count'], cuts=audit.get('cuts'),
        certification=[r['value'] for r in completed if r['event'] == 'certification'],
        probes=[r['value'] for r in completed if r['event'] == 'probe_complete'],
        sa=dict(raw_candidates=sum(s.get('raw_candidates', 0) for s in sampling),
            raw_discrete_feasible=sum(s.get('raw_discrete_feasible', 0) for s in sampling),
            distinct_seeds=len({s['seed'] for s in sampling}),
            proposal_outcomes=dict(Counter(p.get('outcome') for p in proposals if p.get('origin') == 'sa')),
            repair_cache_hits=sum(p.get('repair_cache_hit', False) for p in proposals)),
        llm_calls=final['llm_calls'], receipts=calls, upper_decrease_by_source=decreases,
        reference_cost=cost, reference_cost_difference=final['objective']-cost if cost else None,
        reference_cost_gap=(final['objective']-cost)/abs(cost) if cost else None)
    if valid:
        solution = read(folder/'audit'/audit['final_solution'])
        result['final_solution'] = dict(artifact=str(folder/'audit'/audit['final_solution']),
            fixed_cost=solution['fixed_cost'], expected_fuel_cost=solution['objective']-solution['fixed_cost'],
            scenarios=[{k: s[k] for k in ('name', 'probability', 'objective')} for s in solution['scenarios']])
    save(folder/'summary.json', result)
    return result


def write_report(directory):
    summaries = [r for seed in (7, 11, 19) for arm in ('C', 'D') if (r := summarize_run(directory, arm, seed))]
    reference = read(directory/'preflight/reference/check.json', {})
    save(directory/'summary.json', dict(status=read(directory/'state.json', {}).get('status'),
        runs=summaries, reference=reference, total_formal_budget_seconds=6*BUDGET))
    lines = ['# RTS-24 C/D 改进实验', '', '每组 1800 秒，包含启动、授权、初始化、求解、代理等待和保存。'
        '预检、参考方案重算和独立审计另计时间。每组均独立初始化。', '',
        '| 组/种子 | UB | LB | 认证 gap | 相对参考成本差 | 达标秒数 | 状态 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in summaries:
        f = r['final']
        lines.append(f"| {r['arm']}/{r['seed']} | {f['objective']:.6f} | {f['lower_bound']:.6f} | "
            f"{100*f['relative_gap']:.6f}% | {100*r['reference_cost_gap']:.6f}% | {r['target_seconds']} | "
            f"{r['stop_reason']} / 审计{'通过' if r['audit_passed'] else '待完成'} |")
    if reference:
        lines += ['', f"独立 Gurobi 参考成本 **{reference['objective']:.6f}**，认证下界 "
            f"**{reference['lower_bound']:.6f}**。参考方案仅用于比较，未输入任何正式组。"]
    for r in summaries:
        lines += ['', f"## {r['arm']} / seed {r['seed']}", '',
            f"完整 SA 轮次 {r['complete_rounds']}，协调 {r['coordination_count']} 次，"
            f"正式 LLM 请求 {r['llm_calls']} 次。", '',
            '| 秒 | UB | LB | 认证 gap | 实际检查点秒 |', '|---:|---:|---:|---:|---:|']
        for t, f in r['snapshots'].items():
            if f:
                lines.append(f"| {t} | {f['objective']:.6f} | {f['lower_bound']:.6f} | "
                    f"{100*f['relative_gap']:.6f}% | {f['elapsed_seconds']:.3f} |")
        lines += ['', f"完整统计、分项耗时、候选低估量及回执：[summary.json]({r['arm']}/seed-{r['seed']}/summary.json)。"]
        if r.get('final_solution') and reference:
            solution = r['final_solution']
            lines += ['', '| 成本项 | 本组 | Gurobi 参考 | 本组减参考 |', '|---|---:|---:|---:|']
            comparisons = [('固定及启动费用', solution['fixed_cost'], reference['fixed_cost']),
                ('期望燃料费用', solution['expected_fuel_cost'], reference['objective']-reference['fixed_cost'])]
            comparisons += [(s['name']+' 场景燃料费用', s['objective'], ref['objective'])
                            for s, ref in zip(solution['scenarios'], reference['scenarios'])]
            for name, actual, expected in comparisons:
                lines.append(f'| {name} | {actual:.6f} | {expected:.6f} | {actual-expected:.6f} |')
        lines += ['', '| 阶段 | 累计秒 |', '|---|---:|']
        lines.extend(f'| {name} | {seconds:.6f} |' for name, seconds in r['timings'].items())
        lines += ['', '异步代理时间与数值计算重叠，不能相加作为墙钟时间。', '',
            f"SA 原始候选 {r['sa']['raw_candidates']} 个，原始离散可行 {r['sa']['raw_discrete_feasible']} 个，"
            f"修复缓存命中 {r['sa']['repair_cache_hits']} 次。", '',
            '| LLM 通道 | 状态 | 完成耗时 | 消费时刻 | 数值评估 | 直接 UB 改善 |',
            '|---|---|---:|---:|---|---:|']
        for receipt in r['receipts']:
            lines.append(f"| {receipt['channel']} | {receipt['status']} | {receipt['seconds']} | "
                f"{receipt['consumed_seconds']} | {receipt['numerical_evaluation_completed']} | "
                f"{receipt['direct_incumbent_decrease']} |")
        if not r['llm_calls']:
            lines += ['', '正式 LLM 调用为零。']
        elif not any(x['numerical_evaluation_completed'] for x in r['receipts']):
            lines += ['', '正式请求未完成有效数值评估，无法据此判断提案的增量收益。']
    lines += ['', '## C/D 配对比较', '', '| 种子 | D−C 成本 | D−C 认证 gap（百分点） | C 达标秒 | D 达标秒 |',
              '|---:|---:|---:|---:|---:|']
    for seed in (7, 11, 19):
        pair = {r['arm']: r for r in summaries if r['seed'] == seed and r['audit_passed']}
        if set(pair) == {'C', 'D'}:
            c, d = pair['C'], pair['D']
            lines.append(f"| {seed} | {d['final']['objective']-c['final']['objective']:.6f} | "
                f"{100*(d['final']['relative_gap']-c['final']['relative_gap']):.6f} | {c['target_seconds']} | {d['target_seconds']} |")
    lines += ['', '认证 gap=(UB−LB)/|UB|；参考成本差=(UB−参考成本)/|参考成本|。'
        '报告以实际调用、评估及配对结果判断 LLM 效果。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summaries


def execute(directory, phase, arm='ALL', seed=None):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    if phase == 'report':
        write_report(directory); return 0
    try:
        if not (directory/'protocol.json').exists():
            for name in INPUTS:
                target = directory/'inputs'/name; target.parent.mkdir(exist_ok=True)
                shutil.copy2(RTS24/'inputs'/name, target)
            save(directory/'protocol.json', dict(arms=['C', 'D'], seeds=[7, 11, 19], budget_seconds=BUDGET,
                tolerance=.001, fresh_initialization=True, reference=str(RTS24/'reference/result.json'),
                extension='Both seed7 audits pass and at least one target_met'))
        save(directory/'state.json', dict(status='RUNNING', phase=phase))
        if phase in ('all', 'preflight') and read(directory/'preflight.json', {}).get('status') != 'PASSED':
            if any((directory/a/f'seed-{s}'/'trace.jsonl').exists() for a in ('C', 'D') for s in (7, 11, 19)):
                raise ValueError('Cannot change preflight after formal runs')
            snapshot(directory)
            preflight(directory)
        evidence = read(directory/'preflight.json', {})
        if evidence.get('status') != 'PASSED' or evidence.get('binding') != binding(directory):
            raise ValueError('Preflight missing or stale')
        for run_seed in ((seed,) if seed else (7, 11, 19)) if phase in ('all', 'run', 'audit') else ():
            if run_seed != 7:
                first = [summarize_run(directory, a, 7) for a in ('C', 'D')]
                if not (all(r and r['audit_passed'] for r in first) and any(r['target_met'] for r in first)):
                    save(directory/'extension.json', dict(status='NOT_ELIGIBLE', seed=run_seed))
                    break
            for a in (('C', 'D') if arm == 'ALL' else (arm,)):
                folder = directory/a/f'seed-{run_seed}'
                state = read(folder/'state.json', {})
                if phase in ('all', 'run') and not state.get('numerical_runs_started'):
                    save(folder/'manifest.json', dict(binding=evidence['binding'], arm=a, seed=run_seed,
                        fresh_initialization=True, options=options(directory, a, run_seed, folder)))
                    launch('run', folder, BUDGET, options=options(directory, a, run_seed, folder),
                           estimated_peak_bytes=evidence['estimated_peak_bytes'])
                if (folder/'trace.jsonl').exists() and not read(folder/'audit/audit.json', {}).get('passed'):
                    launch('audit', folder/'audit', 600, run_directory=str(folder), inputs=str(directory/'inputs'))
                write_report(directory)
        save(directory/'state.json', dict(status='FINISHED', phase=phase))
        write_report(directory)
        return 0
    except Stopped as error:
        save(directory/'state.json', dict(status=error.state['status'], stage=error.stage, detail=error.state))
        return 3
    except Exception as error:
        save(directory/'state.json', dict(status='FAILED', error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    raise SystemExit(worker_main(read(parser.parse_args().job)))
