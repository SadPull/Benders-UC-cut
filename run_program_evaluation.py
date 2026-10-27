"""程序验证实验入口：--prepare 训练归档、--run 目录、--audit 目录、--report 目录。

rts24 原工况 5 个种子（600 秒）比较 SA+R、SA+L118、SA+L24r2、SA+L24，负荷扰动工况各 3 个种子（300 秒）比较
SA+R、SA+L118、SA+L24；所有对照方法在本实验中重新冷启动运行。单任务串行、单线程，在线 LLM 调用为零；
参考解只用于事后计分。命令、预算账本、运行工作进程与审计移植自已归档的 run_paper_supplement.py。
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
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

import program_evaluation as study
import offline_pilot
from run_paper_pilot import digest, read, save
from suc_benders.config import load_options
from suc_benders.data import load_data

ROOT = Path(__file__).resolve().parent
L118_SHA256 = 'df7674289f2bef92fd38457a549711c6523b6bb9d6477cbce33ae487bd447373'
# 方法到程序名（文件名去掉 .py），如 SA+L24 -> L24；协议与审计都按程序名记哈希。
PROGRAM_NAMES = {arm: path.stem for arm, path in offline_pilot.PROGRAMS.items()}
ROLE_ARMS = dict(final=study.FINAL_ARM, early=study.EARLY_ARM)
GENERATION_FILES = ('final-program-freeze.json', 'protocol.json', 'manifest.json', 'report.md',
                    'learning-curve.json', 'closed-loop.json', 'state.json')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def source_files():
    return sorted([*ROOT.glob('*.py'), *ROOT.glob('suc_benders/*.py'), *ROOT.glob('programs/*.py'),
                   *ROOT.glob('tests/*.py'), ROOT/'requirements.txt', ROOT/'README.md',
                   *ROOT.glob('data/**/*.json'), *ROOT.glob('instances/**/*.json')])


def smoke_jobs(protocol):
    return [dict(id=f'smoke-{arm}', stage='checks', case=study.CASE, variant='base', instance=f'{study.CASE}-base',
                 arm=arm, seed=17, seconds=30.) for arm in (study.CONTROL_ARM, study.L118_ARM, study.FINAL_ARM)]


def all_jobs(protocol):
    return {j['id']: j for j in smoke_jobs(protocol)+protocol['jobs']}


# 修订（2026-10-26，用户批准；训练结束后、任何测试运行之前）：第 1 轮没有有效程序时，
# 早期程序改取第一个产生有效程序的轮次中验证均分最高的程序，T1 的门槛不变。
def early_program(training, freeze):
    import program_training as lt
    state = read(training/'state.json')
    rows, programs = state['rows'], state['programs']
    scored = {}
    for identifier, program in programs.items():
        value = lt.mean_j(rows, identifier, 'validation', lt.counts()['validation'])
        if program['status'] == 'EVALUATED' and value is not None:
            scored[identifier] = (value, program['round'])
    first = min(r for _, r in scored.values())
    origin = min((i for i, (_, r) in scored.items() if r == first), key=lambda i: (-scored[i][0], i))
    return dict(origin=origin, round=first, validation_mean_J=scored[origin][0], sha256=programs[origin]['sha256'],
                source=training/'programs'/f'{origin}.py', final=freeze['chosen']['final'])


# 早期程序的文件名带轮次（L24r{轮}），必须已在 offline_pilot 中登记为方法。
def early_path(round_index):
    path = offline_pilot.PROGRAMS[study.EARLY_ARM]
    if path.stem != f'L24r{round_index}':
        raise ValueError(f'The early program is from round {round_index}; '
                         f'register SA+L24r{round_index} in offline_pilot first')
    return path


def prepare(training, output=None):
    training = training.resolve()
    freeze = read(training/'final-program-freeze.json')
    if not freeze['gate']['passed'] or read(training/'state.json')['status'] != 'SELECTED':
        raise ValueError('Training did not select a program that passed its gate')
    final = {n: dict(i) for n, i in freeze['final_programs'].items()}
    amendment = None
    paths = dict(final=offline_pilot.PROGRAMS[study.FINAL_ARM])
    if 'early' in final:
        paths['early'] = early_path(1)
    else:
        early = early_program(training, freeze)
        if early['origin'] == early['final']:
            raise ValueError('The earliest valid program is the final program itself; T1 is not applicable')
        paths['early'] = early_path(early['round'])
        if paths['early'].exists():
            if digest(paths['early']) != early['sha256']:
                raise ValueError(paths['early'].name+' exists and differs from the amended early program')
        else:
            shutil.copyfile(early['source'], paths['early'])
        final['early'] = dict(path=paths['early'].relative_to(ROOT).as_posix(), sha256=early['sha256'],
                              origin=early['origin'])
        amendment = dict(made_at=now(), approved_by='user', timing='After training, before any evaluation run',
            reason='Both round-1 programs were STATIC_INVALID, so the registered early program (best round-1 program) '
                   'does not exist',
            change='Early program = validation-best program of the first round that produced valid programs; T1 gate unchanged',
            early_origin=early['origin'], early_round=early['round'], early_validation_mean_J=early['validation_mean_J'])
    for name, info in final.items():
        if digest(paths[name]) != info['sha256']:
            raise ValueError(f'{name} differs from the training freeze')
    if digest(offline_pilot.L118) != L118_SHA256:
        raise ValueError('L118 changed')
    include_early = 'early' in final and final['early']['origin'] != final['final']['origin']
    t1_reason = None if include_early else 'NO_IMPROVEMENT_AFTER_ROUND_1'
    names = {role: PROGRAM_NAMES[ROLE_ARMS[role]] for role in final}
    output = (output or ROOT/'results'/f"program-evaluation-rts24-{datetime.now(timezone(timedelta(hours=8))):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}").resolve()
    if (output/'manifest.json').exists():
        raise FileExistsError('Prepared experiment already exists; never overwrite its snapshot')
    output.mkdir(parents=True, exist_ok=True)
    for p in source_files():
        target = output/'snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    study.write_instances(output/'inputs')
    (output/'generation').mkdir()
    for name in GENERATION_FILES:
        shutil.copyfile(training/name, output/'generation'/name)
    protocol = dict(version=1, created_at=now(), training=str(training),
        programs=dict(L118=L118_SHA256, **{names[n]: i['sha256'] for n, i in final.items()},
                      origins={names[n]: i['origin'] for n, i in final.items()}),
        include_early=include_early, t1_reason=t1_reason, amendment=amendment, jobs=study.jobs(include_early),
        limits_seconds=study.LIMITS, criteria=study.CRITERIA, workers=1, numerical_threads=1, gurobi_threads=1,
        online_llm_calls=0, options='12 h blocks, 8 reads, 16 sweeps; one operator burst per SA round with equal wall time',
        reference_usage='Gurobi 300 s per instance; post-hoc scoring only, the search receives an empty reference',
        pairing='All arms of an instance and seed must share the initial schedule hash; otherwise the run stops',
        arm_order='Rotated within each instance/seed pair', interpretation='Engineering gates, not significance tests',
        test_instances='Byte-identical to the 2026-10-25 supplement inputs (hashes in program_evaluation.TEST_FILES)')
    save(output/'protocol.json', protocol)
    hashes = {p.relative_to(output).as_posix(): digest(p) for directory in ('snapshot', 'inputs', 'generation')
              for p in sorted((output/directory).rglob('*')) if p.is_file()}
    hashes['protocol.json'] = digest(output/'protocol.json')
    save(output/'manifest.json', dict(created_at=now(), files=hashes, python=sys.version, platform=platform.platform(),
        dependencies={n: version(n) for n in ('numpy', 'scipy', 'gurobipy', 'dwave-samplers', 'dimod', 'matplotlib')},
        git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        git_diff=subprocess.run(['git', 'diff', '--stat'], cwd=ROOT, capture_output=True, text=True).stdout))
    save(output/'budget.json', dict(limits=study.LIMITS, records=[]))
    verify(output)
    print(f'PREPARED {output}', flush=True)
    return output


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


def command(output, stage, job, args, seconds):
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
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
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
            artifacts = [output/'runs'/f'{job}.json']
            if job.startswith('reference-'):
                artifacts.append(output/'references'/f'{job[len("reference-"):]}.json')
            entry['artifacts'] = {p.relative_to(output).as_posix(): digest(p) for p in artifacts if p.is_file()}
            entry.update(seconds=perf_counter()-started, ended_at=now())
            save(output/'budget.json', budget)
    print(f"END {job} {entry['status']} elapsed={entry['seconds']:.2f}s", flush=True)
    return entry


def data_for(output, instance):
    p = output/'inputs'/instance
    return load_data(p/'units.json', p/'stochastic.json')


def options_for(output, job):
    folder = output/'inputs'/job['instance']
    return load_options(dict(data_file=str(folder/'units.json'), scenario_file=str(folder/'stochastic.json'),
        seed=job['seed'], time_limit=job['seconds'], block_hours=12, sa_reads=8, sa_sweeps=16,
        max_rounds=10**6, llm_enabled=False, verbose=False), config_path=output/'no-local-config.json')


def trial_worker(output, job):
    verify(output)
    started = perf_counter()
    last_progress = -60.

    def progress(whole, event):
        nonlocal last_progress
        t = perf_counter()-whole.started
        if t-last_progress >= 30 or event in ('initialization', 'final'):
            value = dict(job=job['id'], event=event, seconds=t, objective=whole.result.objective,
                         sa_calls=whole.diagnostics['sa_calls'], updated_at=now())
            save(output/'progress'/f"{job['id']}.json", value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
            last_progress = t
    o = options_for(output, job)
    row = offline_pilot.run_trial(job['instance'], job['arm'], job['seed'], job['seconds'], {},
                                 data=data_for(output, job['instance']), o=o, progress_callback=progress)
    row.update(job, effective_options={k: v for k, v in asdict(o).items() if not k.startswith('llm_')},
               program_sha256=digest(offline_pilot.PROGRAMS[job['arm']]) if job['arm'] in offline_pilot.PROGRAMS else None)
    ref_path = output/'references'/f"{job['instance']}.json"
    row.update(study.score(row, read(ref_path) if ref_path.exists() else {}))
    row['over_budget'] = row['elapsed_seconds'] > job['seconds']
    row['worker_elapsed_seconds'] = perf_counter()-started
    save(output/'runs'/f"{job['id']}.json", row)
    print(f"RESULT {job['id']} status={row['status']} objective={row['objective']} auc={row['auc']}", flush=True)


def reference_worker(output, instance):
    from suc_benders.reference import solve_reference
    verify(output)
    reference = solve_reference(*data_for(output, instance), 300., log_file=output/'logs'/f'{instance}-gurobi.log')
    reference['instance_files'] = {n: digest(output/'inputs'/instance/n) for n in ('units.json', 'stochastic.json')}
    (output/'references').mkdir(exist_ok=True)
    save(output/'references'/f'{instance}.json', reference)
    print(f"REFERENCE {instance} {reference['status']} cost={reference['objective']}", flush=True)


def check_run(output, protocol, job, row):
    if row.get('status') != 'TIME_LIMIT' or not row.get('feasible') or row.get('online_llm_calls') != 0 or row.get('over_budget'):
        raise RuntimeError('Run failed; preserve the attempt and inspect: '+job['id'])
    expected = protocol['programs'].get(PROGRAM_NAMES.get(job['arm']))
    if expected and row.get('program_sha256') != expected:
        raise RuntimeError('Program hash differs from the protocol: '+job['id'])
    if job['stage'] == 'checks':
        return
    for peer in protocol['jobs']:
        path = output/'runs'/f"{peer['id']}.json"
        if peer['instance'] == job['instance'] and peer['seed'] == job['seed'] and peer['id'] != job['id'] and path.exists():
            if read(path).get('initial_u_sha256') != row.get('initial_u_sha256'):
                raise RuntimeError('Initialization differs within a pair: '+job['id'])


def run(output):
    verify(output)
    protocol = read(output/'protocol.json')
    worker = output/'snapshot/run_program_evaluation.py'
    completed = {r['job'] for r in read(output/'budget.json')['records'] if r['status'] == 'COMPLETE'}
    if 'regression' not in completed:
        r = command(output, 'checks', 'regression', [sys.executable, '-m', 'pytest', '-q', 'tests'],
                    min(120., remaining(output, 'checks')))
        if r['status'] != 'COMPLETE':
            raise RuntimeError('Regression tests failed')
    for instance in study.instances():
        jid = 'reference-'+instance
        if jid not in completed:
            command(output, 'references', jid, [sys.executable, str(worker), '--reference-worker', str(output), instance],
                    min(300., remaining(output, 'references')))
    for job in smoke_jobs(protocol)+protocol['jobs']:
        verify(output)
        path = output/'runs'/f"{job['id']}.json"
        if not path.exists():
            entry = command(output, job['stage'], job['id'], [sys.executable, str(worker), '--worker', str(output), job['id']],
                            job['seconds'])
            if not path.exists():
                save(path, dict(job, status=entry['status'], feasible=False, message='Worker did not produce a final result'))
        check_run(output, protocol, job, read(path))
        save(output/'state.json', dict(updated_at=now(), last_completed=job['id'],
             complete=sum((output/'runs'/f"{j['id']}.json").exists() for j in protocol['jobs']), planned=len(protocol['jobs'])))
    entry = command(output, 'audit', 'audit', [sys.executable, str(worker), '--audit-worker', str(output)],
                    min(1100., remaining(output, 'audit')))
    if entry['status'] != 'COMPLETE' or not read(output/'audit.json')['passed']:
        raise RuntimeError('Final audit did not pass')
    entry = command(output, 'audit', 'report', [sys.executable, str(worker), '--report', str(output)],
                    min(60., remaining(output, 'audit')))
    if entry['status'] != 'COMPLETE':
        raise RuntimeError('Report failed')
    save(output/'state.json', dict(status='COMPLETE', complete=len(protocol['jobs']), planned=len(protocol['jobs']), updated_at=now()))
    print('COMPLETE '+str(output), flush=True)


def load_runs(output):
    protocol = read(output/'protocol.json')
    return [read(output/'runs'/f"{j['id']}.json") for j in protocol['jobs'] if (output/'runs'/f"{j['id']}.json").exists()]


def summary_for(output):
    protocol = read(output/'protocol.json')
    rows = load_runs(output)
    if protocol['include_early']:
        return study.summarize(rows), rows
    return study.summarize(rows, include_early=False, t1_reason=protocol['t1_reason']), rows


# 独立重调度每个终点，重算计分（含独立 AUC 积分），并复核配对初始化与在线调用。
def audit(output):
    from suc_benders.benders import dispatch
    from suc_benders.model import model, partition
    from suc_benders.validation import pack, unpack, validate
    verify(output)
    protocol = read(output/'protocol.json')
    rows, checks = load_runs(output), []
    for instance in study.instances():
        d, scenarios = data_for(output, instance)
        m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
        rp = output/'references'/f'{instance}.json'; ref = read(rp) if rp.exists() else {}
        if ref and ref.get('instance_files') != {n: digest(output/'inputs'/instance/n) for n in ('units.json', 'stochastic.json')}:
            raise ValueError('Reference input hashes differ: '+instance)
        for row in (r for r in rows if r['instance'] == instance):
            fixed = dispatch(full, pack(d, np.asarray(row['final_u'], float)), 15.)
            valid = fixed.feasible and validate(d, scenarios, unpack(m, fixed.point))['feasible']
            match = fixed.feasible and abs(fixed.cost-row['objective']) <= row['cost_tolerance']
            recomputed = study.score(row, ref)
            cost = recomputed['reference_cost']; c0 = row['initial']['objective']
            trajectory = [(0., c0)]+[(t, c) for t, c in row['trajectory'] if t >= row['initial']['seconds']]
            if cost not in (None, 0.):
                integral = sum((min(trajectory[i+1][0], row['seconds'])-min(t, row['seconds']))*(c-cost)
                               for i, (t, c) in enumerate(trajectory[:-1]))
                integral += max(0., row['seconds']-trajectory[-1][0])*(trajectory[-1][1]-cost)
                independent = 100*integral/(row['seconds']*abs(cost))
            else:
                independent = None
            scored = all(recomputed[k] == row[k] for k in recomputed)
            scored &= (independent is None and row['auc'] is None or
                       independent is not None and row['auc'] is not None and np.isclose(independent, row['auc'], atol=1e-10))
            checks.append(dict(run=row['id'], feasible=bool(valid), cost_match=bool(match),
                               cost_difference=float(fixed.cost-row['objective']), score_match=bool(scored)))
    summary, _ = summary_for(output)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r['instance'], r['seed']), set()).add(r['initial_u_sha256'])
    same_init = all(len(v) == 1 for v in by_pair.values())
    programs_ok = all(r.get('program_sha256') == protocol['programs'][PROGRAM_NAMES[r['arm']]]
                      for r in rows if r['arm'] in PROGRAM_NAMES)
    result = dict(runs=len(rows), expected=len(protocol['jobs']), solutions=checks, pair_initialization=same_init,
        program_hashes=programs_ok, zero_online_llm_calls=all(r.get('online_llm_calls') == 0 for r in rows), summary=summary,
        passed=len(rows) == len(protocol['jobs']) and same_init and programs_ok
               and all(c['feasible'] and c['cost_match'] and c['score_match'] for c in checks)
               and all(r.get('online_llm_calls') == 0 and not r.get('over_budget') for r in rows))
    save(output/'audit.json', result)
    print(f"AUDIT passed={result['passed']} runs={len(rows)}", flush=True)
    return result


# F6：各配对的终点参考差距；配色与标记取 paper_figures.METHOD_STYLE，写入 folder。
def figure(rows, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_figures import METHOD_STYLE
    names = {study.CONTROL_ARM: 'SA+R (no LLM)', study.L118_ARM: 'SA+L118',
             study.EARLY_ARM: 'SA+L24r2 (early)', study.FINAL_ARM: 'SA+L24 (final)'}
    keys = list(dict.fromkeys((r['instance'], r['seed']) for r in rows))
    arms = [a for a in study.BASE_ARMS if any(r['arm'] == a for r in rows)]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for k, arm in enumerate(arms):
        xs, ys = [], []
        for x, key in enumerate(keys):
            r = next((r for r in rows if (r['instance'], r['seed']) == key and r['arm'] == arm), None)
            if r and r.get('reference_gap_percent') is not None:
                xs.append(x+(k-(len(arms)-1)/2)*.16); ys.append(r['reference_gap_percent'])
        style = METHOD_STYLE[arm]
        ax.scatter(xs, ys, marker=style['marker'], s=30, linewidths=1.2, edgecolors=style['color'],
                   facecolors=style['color'] if style['filled'] else 'none', label=names[arm], zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{i.split('-')[1]}\ns{s}" for i, s in keys], fontsize=7)
    ax.set_ylabel('Final gap to reference (%)'); ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False, fontsize=8, ncol=len(arms), loc='upper center', bbox_to_anchor=(.5, 1.13))
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'F6-program-evaluation.{suffix}', dpi=200)
    plt.close(fig)


def report(output):
    verify(output)
    protocol = read(output/'protocol.json')
    summary, rows = summary_for(output)
    audit_result = read(output/'audit.json') if (output/'audit.json').exists() else dict(passed=None)
    save(output/'summary.json', summary)
    figure(rows, output/'figures')
    pct = lambda v: '—' if v is None else f'{100*v:.1f}%'
    num = lambda v: '—' if v is None else f'{v:.3f}'
    t1, t2, t3 = summary['T1_iteration'], summary['T2_vs_R'], summary['T3_vs_L118']
    lines = ['# 程序验证实验（rts24）', '',
             '训练归档：'+protocol['training'], f"程序来源：{protocol['programs']['origins']}。所有对照方法本实验重新冷启动；在线 LLM 调用为零。", '',
             '## 预登记判据', '']
    if 'wins' in t1:
        lines.append(f"- **T1 迭代效应（SA+L24 对 SA+L24r2，rts24-base）：{t1['status']}**；终点胜出 {t1['wins']}/{t1['pairs']}，AUC 差值中位数 {num(t1['median_auc_difference'])}。")
    else:
        lines.append(f"- **T1 迭代效应：{t1['status']}**。")
    if protocol.get('amendment'):
        a = protocol['amendment']
        lines.append(f"  修订（{a['timing']}，用户批准）：第 1 轮两个程序均未通过静态检查，早期程序改取第 {a['early_round']} 轮验证最佳的 "
                     f"{a['early_origin']}（验证均分 J {a['early_validation_mean_J']:.3f}），门槛不变。")
    lines += [f"- **T2 对无 LLM（SA+L24 对 SA+R，rts24-base）：{t2['status']}**；终点胜出 {t2['wins']}/{t2['pairs']}，AUC 降幅中位数 {pct(t2['median_auc_reduction'])}。",
              f"- **T3 对 L118（SA+L24 对 SA+L118，rts24 全部 {t3['pairs']} 对）：{t3['status']}**；终点胜出 {t3['wins']}/{t3['pairs']}，AUC 差值中位数 {num(t3['median_auc_difference'])}。", '']
    d = summary['descriptive']
    lines.append(f"描述性：SA+L118 对 SA+R（base）终点胜出 {d['L118_vs_R_base']['wins']}/{d['L118_vs_R_base']['pairs']}，"
                 f"AUC 降幅中位数 {pct(d['L118_vs_R_base']['median_auc_reduction'])}；SA+L24 对 SA+R（全部）胜出 "
                 f"{d['L24_vs_R_all']['wins']}/{d['L24_vs_R_all']['pairs']}。")
    if 'L24r2_vs_R_base' in d:
        lines.append(f"SA+L24r2 对 SA+R（base）胜出 {d['L24r2_vs_R_base']['wins']}/{d['L24r2_vs_R_base']['pairs']}，"
                     f"AUC 降幅中位数 {pct(d['L24r2_vs_R_base']['median_auc_reduction'])}。")
    lines += ['', '这些是工程门槛，不是显著性检验。', '', '## 运行', '',
              '| 实例 | 种子 | 方法 | 终点成本 | 参考差距 | AUC |', '|---|---:|---|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: (r['instance'], r['seed'], study.BASE_ARMS.index(r['arm']))):
        gap = r.get('reference_gap_percent')
        lines.append(f"| {r['instance']} | {r['seed']} | {r['arm']} | {r['objective']:.2f} | "
                     f"{'—' if gap is None else f'{gap:.3f}%'} | {num(r.get('auc'))} |")
    budget = read(output/'budget.json')
    lines += ['', '## 审计与预算', '', f"独立审计：{'通过' if audit_result.get('passed') else audit_result.get('passed')}。"]
    for stage, limit in budget['limits'].items():
        used = sum(r['seconds'] for r in budget['records'] if r['stage'] == stage)
        lines.append(f'- {stage}：{used:.1f} / {limit:.0f} 秒。')
    lines += ['', '图：figures/F6-program-evaluation（PNG 与 PDF）。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('REPORT '+str(output/'report.md'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path, metavar='TRAINING')
    for mode in ('run', 'audit', 'report', 'audit-worker'):
        group.add_argument('--'+mode, type=Path)
    group.add_argument('--worker', nargs=2, metavar=('DIRECTORY', 'JOB'))
    group.add_argument('--reference-worker', nargs=2, metavar=('DIRECTORY', 'INSTANCE'))
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
    elif args.worker:
        output = Path(args.worker[0]).resolve()
        trial_worker(output, all_jobs(read(output/'protocol.json'))[args.worker[1]])
    elif args.reference_worker:
        reference_worker(Path(args.reference_worker[0]).resolve(), args.reference_worker[1])
    elif args.audit_worker:
        audit(args.audit_worker.resolve())
    elif args.run:
        run(args.run.resolve())
    elif args.audit:
        output = args.audit.resolve()
        command(output, 'audit', 'audit', [sys.executable, str(output/'snapshot/run_program_evaluation.py'), '--audit-worker', str(output)],
                min(1100., remaining(output, 'audit')))
    else:
        report(args.report.resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())
