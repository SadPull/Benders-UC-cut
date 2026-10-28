"""修复上限实验入口：--prepare 建立冻结目录，--run 目录 依次执行检查、记录运行、离线重放、对照与报告，--report 目录 重写报告。

记录运行：纯 SA（无算子、无 LLM），rts24 种子 17/29/41/53/67 与 ieee118s10 种子 17/29，各 300 秒，单任务串行、单线程；
记录包装器只补记原始全天候选，不改变求解行为（identity 检查）。离线重放比较现有 repair() 与精确最小改动修复 P0；
Gurobi 只作诊断标尺。判据只看 rts24 的 SA 样本，写入 protocol.json 后才运行。
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

import offline_pilot
import repair_headroom as rh
from suc_benders.config import load_options
from suc_benders.data import jsonable, load_data, save_json
from suc_benders.model import model, partition

ROOT = Path(__file__).resolve().parent
INSTANCES = ('rts24', 'ieee118s10')
SEEDS = dict(rts24=(17, 29, 41, 53, 67), ieee118s10=(17, 29))
SECONDS = 300.
LOG_MARGIN = 90.
SAMPLES = dict(rts24=120, ieee118s10=60)
POOL_SAMPLES = 20
SLACK_SAMPLES = dict(rts24=40, ieee118s10=0)
ORACLE_SECONDS = 1500.
ORACLE_RESERVE = 95.
PHASE1, PHASE2 = 30., 60.
CONTROL_SECONDS = 120.
BALL_RADIUS = 2
LIMITS = dict(checks=600., logging=2700., oracle=10800., controls=900., report=120.)
CRITERIA = dict(instance='rts24', origin='sa', seeds=5, pass_seeds=3, pass_rate=.01, fail_seeds=1,
                max_unresolved=.10, min_replayed=60, rel_tolerance=rh.REL_TOLERANCE)
REFERENCES = ROOT/'results'/'paper-reference-20261025-124811-c6f41da2'
SNAPSHOT = ('repair_headroom.py', 'run_repair_headroom.py', 'offline_pilot.py', 'offline_worker.py', 'requirements.txt',
            'tests/conftest.py', 'tests/test_repair_headroom.py')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, jsonable(value))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 显式快照清单：不含有未提交修改的 run_program_evaluation.py、本地配置和根目录杂项文件。
def source_files():
    return sorted([*(ROOT/name for name in SNAPSHOT), *ROOT.glob('suc_benders/*.py'), *ROOT.glob('programs/*.py'),
                   *(ROOT/'instances'/name/f for name in INSTANCES for f in ('units.json', 'stochastic.json'))])


def jobs():
    log = [dict(id=f'log-{i}-s{s}', stage='logging', instance=i, seed=s, seconds=SECONDS, limit=SECONDS+LOG_MARGIN)
           for i in INSTANCES for s in SEEDS[i]]
    oracle = [dict(id=f'oracle-{i}-s{s}', stage='oracle', instance=i, seed=s, limit=ORACLE_SECONDS)
              for i in INSTANCES for s in SEEDS[i]]
    return log, oracle


def prepare(output=None):
    output = (output or ROOT/'results'/f"repair-headroom-{datetime.now(timezone(timedelta(hours=8))):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}").resolve()
    if (output/'manifest.json').exists():
        raise FileExistsError('Prepared experiment already exists; never overwrite its snapshot')
    output.mkdir(parents=True, exist_ok=True)
    for p in source_files():
        target = output/'snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    for name in INSTANCES:
        reference = read(REFERENCES/f'{name}.json')
        for relative, expected in reference['instance_files'].items():
            if digest(ROOT/relative) != expected:
                raise ValueError(f'Reference {name} was solved on different inputs: {relative}')
        for f in ('units.json', 'stochastic.json'):
            target = output/'inputs'/name/f
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/'instances'/name/f, target)
        save(output/'inputs'/'references'/f'{name}.json',
             dict(status=reference['status'], objective=reference['objective'],
                  gurobi_gap_percent=reference.get('gurobi_gap_percent'), source=(REFERENCES/f'{name}.json').relative_to(ROOT).as_posix(),
                  instance_files=reference['instance_files']))
    log, oracle = jobs()
    protocol = dict(version=1, created_at=now(), question='If repair were ideal, would plain-SA samples become improvements?',
        arm='SA', online_llm_calls=0, instances=INSTANCES, seeds=SEEDS, seconds=SECONDS,
        options=dict(block_hours=12, sa_reads=8, sa_sweeps=16, max_rounds=10**6, llm_enabled=False),
        samples=SAMPLES, pool_samples=POOL_SAMPLES, slack_samples=SLACK_SAMPLES, select_stream=rh.SELECT_STREAM,
        oracle=dict(phase1_seconds=PHASE1, phase2_seconds=PHASE2, job_seconds=ORACLE_SECONDS, reserve_seconds=ORACLE_RESERVE,
                    definition='P0 = lexicographic minimal-change repair: min Hamming(u, raw) over the full model, then min true '
                               'cost within Hamming <= h* (+slack); rounded u is dispatched and independently validated'),
        controls=dict(block_ceiling='Min cost with u outside one 12-h block fixed to the incumbent', ball_radius=BALL_RADIUS,
                      seconds=CONTROL_SECONDS),
        criteria=CRITERIA, decisions=dict(
            PASS='>= 3/5 rts24 seeds with >= 1 improving P0 sample AND pooled improving rate >= 1%',
            FAIL='<= 1/5 seeds with an improving sample AND unresolved <= 10% AND every seed replayed >= 60 samples',
            INCONCLUSIVE='Otherwise'),
        denominator='Replayed unique SA samples that differ from the incumbent at sampling time',
        validity=['Incumbent re-dispatch matches the logged objective within 1e-6 relative',
                  'Batch anchor cost equals the incumbent at prepare time for every SA sample',
                  'If the raw sample is dispatch-feasible, P0 has h*=0 and equals the current-repair cost',
                  'Wrapper identity run: identical final_u, sa_calls, outcomes and evaluations with and without logging',
                  'Recomputed repair() reproduces the logged commitment_id and hamming_u'],
        interpretation=['C_b >= incumbent: block coordinate search is exhausted; repair is irrelevant',
                        'C_b < incumbent and P0 improves: repair is a bottleneck',
                        'C_b < incumbent and P0 does not improve: the SA samples themselves (QUBO cost proxy) are the problem'],
        reference_usage='Gurobi references are only used to express costs as reference-gap points',
        jobs=log+oracle, limits_seconds=LIMITS, workers=1, numerical_threads=1, gurobi_threads=1)
    save(output/'protocol.json', protocol)
    hashes = {p.relative_to(output).as_posix(): digest(p) for directory in ('snapshot', 'inputs')
              for p in sorted((output/directory).rglob('*')) if p.is_file()}
    hashes['protocol.json'] = digest(output/'protocol.json')
    save(output/'manifest.json', dict(created_at=now(), files=hashes, python=sys.version, platform=platform.platform(),
        dependencies={n: version(n) for n in ('numpy', 'scipy', 'gurobipy', 'dwave-samplers', 'dimod')},
        git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        git_status=subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True).stdout))
    save(output/'budget.json', dict(limits=LIMITS, records=[]))
    verify(output)
    print(f'PREPARED {output}', flush=True)
    return output


# 冻结文件不得改动；只有已完成任务的产物登记哈希（中断的重放会续写，不登记）。
def verify(output):
    manifest = read(output/'manifest.json')
    changed = [name for name, h in manifest['files'].items() if not (output/name).is_file() or digest(output/name) != h]
    if changed:
        raise RuntimeError('Frozen files changed: '+', '.join(changed[:8]))
    for record in read(output/'budget.json')['records']:
        for name, expected in record.get('artifacts', {}).items():
            if not (output/name).is_file() or digest(output/name) != expected:
                raise RuntimeError('Recorded artifact changed: '+name)


def remaining(output, stage):
    budget = read(output/'budget.json')
    used = sum(r['seconds'] for r in budget['records'] if r['stage'] == stage)
    total = sum(r['seconds'] for r in budget['records'])
    return max(0., min(budget['limits'][stage]-used, sum(budget['limits'].values())-total))


# 子进程单线程运行并计入分阶段预算；超时用 taskkill 结束整个进程树。
def command(output, stage, job, args, seconds, artifacts=(), cwd=ROOT):
    if remaining(output, stage) < seconds:
        raise RuntimeError(f'BUDGET_INSUFFICIENT: {stage}/{job}')
    budget = read(output/'budget.json')
    if any(r['status'] == 'RUNNING' for r in budget['records']):
        raise RuntimeError('An earlier child is still marked RUNNING; inspect it before continuing')
    entry = dict(stage=stage, job=job, seconds=0., status='RUNNING', started_at=now(), limit_seconds=seconds)
    budget['records'].append(entry); save(output/'budget.json', budget)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    log = output/'logs'/f'{job}.log'; log.parent.mkdir(exist_ok=True)
    started = perf_counter()
    print(f'START {stage} {job} limit={seconds:.0f}s', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        p = subprocess.Popen(args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        entry['pid'] = p.pid; save(output/'budget.json', budget)
        try:
            code = p.wait(timeout=seconds+5.)
            entry['status'], entry['exit_code'] = ('COMPLETE' if code == 0 else 'FAILED'), code
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            else:
                p.kill()
            p.wait(); entry['status'] = 'PROCESS_TIMEOUT'
        finally:
            if entry['status'] == 'COMPLETE':
                entry['artifacts'] = {name: digest(output/name) for name in artifacts if (output/name).is_file()}
            entry.update(seconds=perf_counter()-started, ended_at=now())
            save(output/'budget.json', budget)
    print(f"END {job} {entry['status']} elapsed={entry['seconds']:.2f}s", flush=True)
    return entry


def data_for(output, instance):
    p = output/'inputs'/instance
    return load_data(p/'units.json', p/'stochastic.json')


def options_for(output, instance, seed, seconds):
    folder = output/'inputs'/instance
    return load_options(dict(data_file=str(folder/'units.json'), scenario_file=str(folder/'stochastic.json'),
        seed=seed, time_limit=seconds, block_hours=12, sa_reads=8, sa_sweeps=16,
        max_rounds=10**6, llm_enabled=False, verbose=False), config_path=output/'no-local-config.json')


def find(output, job_id):
    log, oracle = jobs()
    return next(j for j in log+oracle if j['id'] == job_id)


# 同一 rts24 单轮运行分别不带、带记录包装器各跑一次，方案、调用数、提议结局与评估序列须完全一致。
def identity_worker(output):
    verify(output)
    data = data_for(output, 'rts24')
    runs = []
    for logged in (False, True):
        kept = {}
        o = options_for(output, 'rts24', 17, 120.)
        keep = lambda whole, event: kept.setdefault('whole', whole)
        if logged:
            with rh.log_proposals():
                row = offline_pilot.run_trial('rts24', 'SA', 17, 120., {}, rounds=1, data=data, o=o, progress_callback=keep)
        else:
            row = offline_pilot.run_trial('rts24', 'SA', 17, 120., {}, rounds=1, data=data, o=o, progress_callback=keep)
        whole = kept['whole']
        runs.append(dict(status=row['status'], final_u=row['final_u'], sa_calls=row['sa_calls'],
            outcomes=[(r.get('proposal_id'), r.get('outcome')) for r in whole.diagnostics['proposals']],
            evaluations=[(e['source'], e['commitment_hash'], e['objective']) for e in whole.diagnostics['evaluations']],
            logged_fields=sum('raw_bits' in r for r in whole.diagnostics['proposals'])))
    keys = ('status', 'final_u', 'sa_calls', 'outcomes', 'evaluations')
    result = dict(identical=all(runs[0][k] == runs[1][k] for k in keys) and runs[0]['status'] == 'ROUNDS',
                  restored=rh.search_core.WholeDay.prepare.__qualname__ == 'WholeDay.prepare',
                  proposals=len(runs[0]['outcomes']), evaluations=len(runs[0]['evaluations']),
                  logged_fields=runs[1]['logged_fields'], sa_calls=runs[0]['sa_calls'])
    save(output/'runs'/'identity.json', result)
    print('IDENTITY', json.dumps(result), flush=True)


def log_worker(output, job):
    verify(output)
    started = perf_counter()
    kept, last = {}, [-60.]

    def progress(whole, event):
        kept['whole'] = whole
        t = perf_counter()-whole.started
        if t-last[0] >= 30 or event == 'initialization':
            value = dict(job=job['id'], event=event, seconds=t, objective=whole.result.objective,
                         sa_calls=whole.diagnostics['sa_calls'], proposals=len(whole.diagnostics['proposals']), updated_at=now())
            save(output/'progress'/f"{job['id']}.json", value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
            last[0] = t
    o = options_for(output, job['instance'], job['seed'], job['seconds'])
    with rh.log_proposals() as holder:
        row = offline_pilot.run_trial(job['instance'], 'SA', job['seed'], job['seconds'], {},
                                     data=data_for(output, job['instance']), o=o, progress_callback=progress)
    whole = kept['whole']
    rows = rh.collect(whole)
    samples = f"samples/{job['instance']}-s{job['seed']}.jsonl.gz"
    rh.write_jsonl_gz(output/samples, rows)
    row.update(job, effective_options={k: v for k, v in asdict(o).items() if not k.startswith('llm_')},
               incumbents=holder.incumbents, search_statistics=whole.search_statistics(), samples_file=samples,
               logged_proposals=len(rows), over_budget=row['elapsed_seconds'] > job['seconds'],
               worker_elapsed_seconds=perf_counter()-started)
    save(output/'runs'/f"{job['id']}.json", row)
    print(f"RESULT {job['id']} status={row['status']} objective={row['objective']} logged={len(rows)}", flush=True)


def check_log(row):
    if row.get('status') != 'TIME_LIMIT' or not row.get('feasible') or row.get('online_llm_calls') != 0 or row.get('over_budget'):
        raise RuntimeError('Logging run failed; preserve the attempt and inspect: '+row.get('id', '?'))


def _header(output, job, m, full, rows, incumbents, shape):
    path = output/'oracle'/f"{job['instance']}-s{job['seed']}-header.json"
    if path.exists():
        return read(path)
    used = sorted({r['incumbent_id'] for r in rows if r['origin'] in ('sa', 'pool')})
    redispatch = {}
    for identity in used:
        info = incumbents[identity]
        result = rh.evaluate_u(m, full, rh.bits_to_u(info['bits'], shape))
        redispatch[identity] = dict(logged=info['objective'], recomputed=result['objective'], ok=bool(
            result.get('feasible') and abs(result['objective']-info['objective']) <= 1e-6*abs(info['objective'])))
    anchors = [r for r in rows if r['origin'] == 'sa' and r.get('anchor_cost') is not None]
    mismatched = sum(abs(r['anchor_cost']-r['incumbent_objective']) > 1e-9*abs(r['incumbent_objective']) for r in anchors)
    sa, sa_info = rh.select(rows, incumbents, shape, SAMPLES[job['instance']], job['seed'])
    pool, pool_info = rh.select(rows, incumbents, shape, POOL_SAMPLES, job['seed'], origin='pool')
    header = dict(job=job['id'], incumbents_used=len(used), redispatch=redispatch, anchor_checked=len(anchors),
                  anchor_mismatches=mismatched, sa=sa_info, pool=pool_info,
                  sa_selected=[r['raw_sha256'] for r in sa], pool_selected=[r['raw_sha256'] for r in pool],
                  slack_selected=[r['raw_sha256'] for r in sa[:SLACK_SAMPLES[job['instance']]]])
    save(path, header)
    return header


# 离线重放：每个样本一行写入并立即落盘，可续跑；剩余不足 ORACLE_RESERVE 秒时不再开始新样本。
def oracle_worker(output, job):
    verify(output)
    started = perf_counter()
    deadline = started+job['limit']-30.
    instance, seed = job['instance'], job['seed']
    d, scenarios = data_for(output, instance)
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    hours = {j+1: np.asarray(b.times) for j, b in enumerate(partition(m, 12)[0])}
    log = read(output/'runs'/f'log-{instance}-s{seed}.json')
    rows = rh.read_jsonl_gz(output/log['samples_file'])
    incumbents, shape = log['incumbents'], (d.n_units, d.horizon)
    header = _header(output, job, m, full, rows, incumbents, shape)
    by_hash = {}
    for r in rows:
        by_hash.setdefault((r['origin'], r['raw_sha256']), r)
    counts = {}
    for r in rows:
        counts[(r['origin'], r['raw_sha256'])] = counts.get((r['origin'], r['raw_sha256']), 0)+1
    tasks = ([('sa', h, 0) for h in header['sa_selected']] + [('pool', h, 0) for h in header['pool_selected']]
             + [('slack2', h, 2) for h in header['slack_selected']])
    path = output/'oracle'/f'{instance}-s{seed}.jsonl'
    done = set()
    if path.exists():
        done = {(r['kind'], r['raw_sha256']) for r in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if r}
    stopped = False
    with path.open('a', encoding='utf-8') as stream:
        for kind, h, slack in tasks:
            if (kind, h) in done:
                continue
            if deadline-perf_counter() < ORACLE_RESERVE:
                stopped = True
                break
            r = by_hash[('pool' if kind == 'pool' else 'sa', h)]
            raw = rh.bits_to_u(r['raw_bits'], shape)
            info = incumbents[r['incumbent_id']]
            incumbent, incumbent_u = info['objective'], rh.bits_to_u(info['bits'], shape)
            current = None if kind == 'slack2' else rh.current_repair(m, full, raw)
            p0 = rh.minimal_change(m, full, raw, slack=slack, phase1=PHASE1, phase2=PHASE2)
            outside = np.setdiff1d(np.arange(d.horizon), hours[r['block']])
            line = dict(kind=kind, raw_sha256=h, proposal_id=r['proposal_id'], block=r['block'], round=r['round'],
                        multiplicity=counts[('pool' if kind == 'pool' else 'sa', h)], logged_outcome=r['outcome'],
                        logged_evaluation=r['evaluation'], raw_discrete_feasible=r['raw_discrete_feasible'],
                        incumbent=incumbent, hamming_incumbent=int(np.count_nonzero(raw != incumbent_u)),
                        current=current, p0=p0, **{'class': rh.classify(p0, incumbent)},
                        p0_outside_block_changes=None if 'u_bits' not in p0 else int(np.count_nonzero(
                            rh.bits_to_u(p0['u_bits'], shape)[:, outside] != raw[:, outside])))
            if current is not None:
                line['checks'] = dict(
                    logged_match=current.get('commitment_id') == r.get('commitment_id')
                                 and (current.get('hamming') == r.get('hamming_u')),
                    shortcut=None if not (current.get('feasible') and current.get('hamming') == 0) else bool(
                        p0.get('h_star') == 0 and p0.get('objective') is not None
                        and abs(p0['objective']-current['objective']) <= 1e-6*abs(current['objective'])))
            stream.write(json.dumps(jsonable(line), ensure_ascii=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
            print(f"{kind} {h[:10]} class={line['class']} h*={p0.get('h_star')} p0={p0.get('objective')} "
                  f"cur={None if current is None else current.get('objective')}", flush=True)
    save(output/'oracle'/f'{instance}-s{seed}-done.json', dict(job=job['id'], planned=len(tasks), stopped_early=stopped,
         completed=len(path.read_text(encoding='utf-8').splitlines()), seconds=perf_counter()-started, updated_at=now()))


# 对照：每个算例的现任方案（所有种子共用的初始方案）在各块的块内上限，以及 Hamming 半径 BALL_RADIUS 的球内最优。
def controls_worker(output):
    verify(output)
    result = {}
    for instance in INSTANCES:
        d, scenarios = data_for(output, instance)
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        initial = [read(output/'runs'/f'log-{instance}-s{s}.json')['initial'] for s in SEEDS[instance]]
        if any(i['u'] != initial[0]['u'] for i in initial):
            raise RuntimeError(f'Initial schedules differ across seeds on {instance}')
        anchor = np.asarray(initial[0]['u'], dtype=float)
        incumbent = rh.evaluate_u(m, full, anchor)
        blocks = {}
        for j, b in enumerate(partition(m, 12)[0]):
            blocks[str(j+1)] = rh.block_ceiling(m, full, anchor, np.asarray(b.times), CONTROL_SECONDS)
            print(instance, 'block', j+1, blocks[str(j+1)].get('status'), blocks[str(j+1)].get('objective'), flush=True)
        sphere = rh.ball(m, full, anchor, BALL_RADIUS, CONTROL_SECONDS)
        print(instance, 'ball', sphere.get('status'), sphere.get('objective'), flush=True)
        result[instance] = dict(incumbent=incumbent, block_ceiling=blocks, ball=dict(radius=BALL_RADIUS, **sphere))
    save(output/'runs'/'controls.json', result)


def oracle_lines(output, instance, seed):
    path = output/'oracle'/f'{instance}-s{seed}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []


def summarize(output):
    protocol = read(output/'protocol.json')
    reference = {i: read(output/'inputs'/'references'/f'{i}.json')['objective'] for i in INSTANCES}
    identity = read(output/'runs'/'identity.json') if (output/'runs'/'identity.json').exists() else {}
    controls = read(output/'runs'/'controls.json') if (output/'runs'/'controls.json').exists() else {}
    runs, classes, validity = {}, {}, dict(identity=bool(identity.get('identical') and identity.get('restored')))
    problems = []
    for instance in INSTANCES:
        for seed in SEEDS[instance]:
            name = f'{instance}-s{seed}'
            header_path = output/'oracle'/f'{name}-header.json'
            if not header_path.exists():
                continue
            header, lines = read(header_path), oracle_lines(output, instance, seed)
            log = read(output/'runs'/f'log-{name}.json')
            incumbent = log['initial']['objective']
            sa = [r for r in lines if r['kind'] == 'sa']
            if not all(v['ok'] for v in header['redispatch'].values()):
                problems.append(f'{name}: incumbent re-dispatch mismatch')
            if header['anchor_mismatches']:
                problems.append(f"{name}: {header['anchor_mismatches']} anchor mismatches")
            if header['incumbents_used'] != 1:
                problems.append(f"{name}: {header['incumbents_used']} incumbents during SA")
            checked = [r for r in lines if 'checks' in r]
            if not all(r['checks']['logged_match'] for r in checked):
                problems.append(f"{name}: recomputed repair differs from the log")
            if any(r['checks']['shortcut'] is False for r in checked):
                problems.append(f"{name}: P0 differs from current repair on a dispatch-feasible raw sample")
            if instance == CRITERIA['instance']:
                classes[seed] = [r['class'] for r in sa]
            slack = [r for r in lines if r['kind'] == 'slack2']
            pool = [r for r in lines if r['kind'] == 'pool']
            runs[name] = dict(instance=instance, seed=seed, incumbent=incumbent, sa_calls=log['sa_calls'],
                logged_proposals=log['logged_proposals'], selection=header['sa'],
                search_statistics_sa=log['search_statistics'].get('sa', {}).get('outcomes'),
                describe=rh.describe(lines, incumbent, reference[instance]),
                pool=dict(replayed=len(pool), improving=sum(r['class'] == 'improving' for r in pool)),
                slack2=dict(replayed=len(slack), **{c: sum(r['class'] == c for r in slack)
                                                    for c in ('improving', 'non_improving', 'unresolved')}),
                anomalies=dict(p0_errors=sum(r['p0'].get('status') == 'ERROR' for r in lines),
                               p0_infeasible=sum(r['p0'].get('feasible') is False for r in lines)))
    validity['checks'] = not problems
    decision = rh.gate(classes, CRITERIA) if classes else dict(decision='NOT_RUN')
    valid = all(validity.values())
    return dict(updated_at=now(), decision=decision['decision'] if valid else 'INVALID', gate=decision,
                validity=validity, problems=problems, identity=identity, controls=controls, runs=runs,
                reference=reference, criteria=protocol['criteria'])


def _f(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def report(output):
    s = summarize(output)
    save(output/'summary.json', s)
    lines = ['# 修复上限实验报告', '', f"生成时间：{s['updated_at']}", '',
             f"**判据结论：{s['decision']}**（rts24 SA 样本；判据：{s['criteria']}）", '']
    g = s['gate']
    if 'replayed' in g:
        lines += [f"- 有改进样本的种子：{g['seeds_with_improving']}/{len(g['replayed'])}；合并改进率 {100*g['improving_rate']:.2f}%"
                  f"（{sum(g['improving'].values())}/{g['total']}）；unresolved 占比 {100*g['unresolved_share']:.1f}%", '']
    lines += [f"有效性：{s['validity']}；问题：{s['problems'] or '无'}", '',
              '| 运行 | SA 调用 | 记录提议 | 去重/与现任相同/可选 | 重放 | improving | non_improving | unresolved | 现有修复改进 | 最佳 P0 增益 % | 现任差距 % | 最佳 P0 差距 % |',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in s['runs'].items():
        dsc, sel = r['describe'], r['selection']
        lines.append(f"| {name} | {r['sa_calls']} | {r['logged_proposals']} | {sel['unique']}/{sel['identical_to_incumbent']}/{sel['eligible']} "
                     f"| {dsc['replayed']} | {dsc['classes']['improving']} | {dsc['classes']['non_improving']} | {dsc['classes']['unresolved']} "
                     f"| {dsc['current_improving']} | {_f(dsc['best_p0_gain_percent'])} | {_f(dsc['incumbent_gap_percent'])} | {_f(dsc['best_p0_gap_percent'])} |")
    lines += ['', '| 运行 | 原始离散可行 | 现有修复可行 | P0 可行 | P0 救回 | h_cur 中位 | h* 中位 | 距现任中位 | 修复代价中位 | P0 相对现任中位 | 现有修复相对现任中位 | P0 块外改动>0 | 加权改进率 | 早/中/晚改进 | Δ=2 改进 | pool 改进 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for name, r in s['runs'].items():
        dsc = r['describe']
        period = '/'.join(f"{v['improving']}:{v['replayed']}" for v in dsc['by_period'].values())
        lines.append(f"| {name} | {dsc['raw_discrete_feasible']} | {dsc['current_feasible']} | {dsc['p0_feasible']} | {dsc['rescued']} "
                     f"| {_f(dsc['median_h_current'], 1)} | {_f(dsc['median_h_star'], 1)} | {_f(dsc['median_hamming_to_incumbent'], 1)} "
                     f"| {_f(dsc['median_repair_tax'], 1)} | {_f(dsc['median_relative_to_incumbent_p0'], 5)} "
                     f"| {_f(dsc['median_relative_to_incumbent_current'], 5)} | {dsc['outside_block_changes']['positive']} "
                     f"| {100*dsc['weighted_improving_rate']:.2f}% | {period} | {r['slack2']['improving']}/{r['slack2']['replayed']} "
                     f"| {r['pool']['improving']}/{r['pool']['replayed']} |")
    if s['controls']:
        lines += ['', '## 对照（只作描述）', '', '| 算例 | 现任成本 | 块 | 块内上限 C_b | 状态 | 相对现任 % |', '|---|---:|---|---:|---|---:|']
        for instance, c in s['controls'].items():
            base = c['incumbent']['objective']
            for block, b in c['block_ceiling'].items():
                value = b.get('objective') if b.get('feasible') else None
                lines.append(f"| {instance} | {_f(base, 2)} | {block} | {_f(value, 2)} | {b['status']} "
                             f"| {_f(None if value is None else 100*(value-base)/abs(base), 4)} |")
            ball = c['ball']
            value = ball.get('objective') if ball.get('feasible') else None
            lines.append(f"| {instance} | {_f(base, 2)} | 球 r={ball['radius']} | {_f(value, 2)} | {ball['status']} "
                         f"| {_f(None if value is None else 100*(value-base)/abs(base), 4)} |")
    lines += ['', '解读规则（预登记）：', *[f'- {t}' for t in read(output/'protocol.json')['interpretation']], '',
              'Gurobi 只作诊断标尺；对照方法仍只有无 LLM 的 SA。改进与否以独立调度校验后的成本为准。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"REPORT {s['decision']}", flush=True)


def run(output):
    verify(output)
    worker = str(output/'snapshot'/'run_repair_headroom.py')
    completed = lambda: {r['job'] for r in read(output/'budget.json')['records'] if r['status'] == 'COMPLETE'}
    if 'tests' not in completed():
        r = command(output, 'checks', 'tests', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    'tests/test_repair_headroom.py'], 300., cwd=output/'snapshot')
        if r['status'] != 'COMPLETE':
            raise RuntimeError('Experiment tests failed')
    if 'identity' not in completed():
        command(output, 'checks', 'identity', [sys.executable, worker, '--identity-worker', str(output)], 240.,
                artifacts=('runs/identity.json',))
    identity = read(output/'runs'/'identity.json')
    if not (identity['identical'] and identity['restored'] and identity['logged_fields'] > 0):
        raise RuntimeError('Logging wrapper changed the search or did not log; stop')
    log, oracle = jobs()
    for job in log:
        verify(output)
        if job['id'] not in completed():
            command(output, 'logging', job['id'], [sys.executable, worker, '--log-worker', str(output), job['id']], job['limit'],
                    artifacts=(f"runs/{job['id']}.json", f"samples/{job['instance']}-s{job['seed']}.jsonl.gz"))
        if not (output/'runs'/f"{job['id']}.json").exists():
            raise RuntimeError('Logging worker did not produce a result: '+job['id'])
        check_log(read(output/'runs'/f"{job['id']}.json"))
    for job in oracle:
        verify(output)
        if job['id'] not in completed():
            name = f"{job['instance']}-s{job['seed']}"
            command(output, 'oracle', job['id'], [sys.executable, worker, '--oracle-worker', str(output), job['id']], job['limit'],
                    artifacts=(f'oracle/{name}.jsonl', f'oracle/{name}-header.json', f'oracle/{name}-done.json'))
    if 'controls' not in completed():
        command(output, 'controls', 'controls', [sys.executable, worker, '--controls-worker', str(output)],
                min(LIMITS['controls'], remaining(output, 'controls')), artifacts=('runs/controls.json',))
    entry = command(output, 'report', 'report', [sys.executable, worker, '--report', str(output)], 60.)
    if entry['status'] != 'COMPLETE':
        raise RuntimeError('Report failed')
    print('COMPLETE '+str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    for mode in ('run', 'report', 'identity-worker', 'controls-worker'):
        group.add_argument('--'+mode, type=Path)
    group.add_argument('--log-worker', nargs=2, metavar=('DIRECTORY', 'JOB'))
    group.add_argument('--oracle-worker', nargs=2, metavar=('DIRECTORY', 'JOB'))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run(args.run.resolve())
    elif args.report:
        report(args.report.resolve())
    elif args.identity_worker:
        identity_worker(args.identity_worker.resolve())
    elif args.controls_worker:
        controls_worker(args.controls_worker.resolve())
    elif args.log_worker:
        output = Path(args.log_worker[0]).resolve()
        log_worker(output, find(output, args.log_worker[1]))
    else:
        output = Path(args.oracle_worker[0]).resolve()
        oracle_worker(output, find(output, args.oracle_worker[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
