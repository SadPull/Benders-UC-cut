"""论文补充实验的驱动：参考解、冒烟、对比、案例、报告与复核。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np

import paper_pilot as pilot
from suc_benders.benders import dispatch
from suc_benders.data import jsonable, save_json
from suc_benders.model import model, partition
from suc_benders.reference import solve_reference
from suc_benders.validation import pack, unpack, validate

ROOT = pilot.ROOT
REFERENCE_SECONDS = 3600.
COMPARE = ('ieee118s10', 'rts24')
SEEDS = (17, 29, 41, 53, 67)
SECONDS, WORKERS = 600., 5
SMOKE_SECONDS = 90.
# T3：v0.35.0 在线 LLM 的已有成对运行（单种子、旧版流程），按归档目录取数。
ONLINE = ROOT/'archives/v0.35.0.json'
ONLINE_PAIRS = (('RCUC20, 3 h blocks', 'manual-seed17-20261020-143502-f8585936', 'manual-seed17-20261020-150525-7e05de22'),
                ('RCUC20, 12 h blocks', 'manual-seed17-20261020-183542-c83b9824', 'manual-seed17-20261020-190541-0db3e34f'),
                ('classic10', 'manual-seed17-20261020-195901-012e563c', 'manual-seed17-20261020-195901-2174926b'),
                ('RTS24, 1800 s', 'manual-seed17-20261020-202903-bf25b4b5', 'manual-seed17-20261020-202903-72a40eae'),
                ('RTS24, 5400 s', 'manual-seed17-20261020-210949-7ff40209', 'manual-seed17-20261020-210949-b40f0800'),
                ('IEEE118', 'manual-seed17-20261020-224136-14459f84', 'manual-seed17-20261020-224136-7326d548'))
SOURCES = ('paper_pilot.py', 'paper_figures.py', 'run_paper_pilot.py', 'make_ieee118_scenarios.py',
           'offline_pilot.py', 'offline_worker.py', 'programs/L3.py', 'tests/test_offline_pilot.py',
           'tests/test_paper_pilot.py')
CRITERIA = {
    'M1 ieee118s10': f'L+SA 终点差距严格小于 A 的种子 >= {pilot.WINS_NEEDED}/{len(SEEDS)}，且 AUC 降幅中位数 >= '
                     f'{pilot.AUC_REDUCTION:.0%}',
    'M2 rts24': '同 M1，单独报告'}
DECISIONS = {
    (True, True): 'M1 通过：ieee118s10 作为论文主随机算例，题目保留"随机"',
    (True, False): 'M1 通过：ieee118s10 作为论文主随机算例，题目保留"随机"（RTS24 未通过，如实报告）',
    (False, True): 'M1 未通过、M2 通过：RTS24 作为主随机算例，单场景 IEEE118 作为规模算例',
    (False, False): 'M1、M2 都未通过：停下汇报，由用户决定去掉"随机"或离线重新生成能处理场景的程序'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, jsonable(value))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_tests(output):
    with (output/'tests.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError('Required tests failed; see tests.log')


def stamp(kind):
    return ROOT/'results'/f"paper-{kind}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def instance_hashes(name):
    folder = pilot.INSTANCES[name]
    return {f'{folder.relative_to(ROOT).as_posix()}/{n}': digest(folder/n) for n in ('units.json', 'stochastic.json')}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout


# 参考解单独求解并缓存到同一目录，记录实例哈希供使用时核对；只当差距尺子。
def run_reference(names, seconds):
    output = stamp('reference')
    save(output/'manifest.json', dict(created_at=datetime.now().astimezone().isoformat(), instances=names,
         seconds=seconds, git_head=git('rev-parse', 'HEAD').strip(),
         files={k: v for name in names for k, v in instance_hashes(name).items()}))
    for name in names:
        d, scenarios = pilot.instance(name)
        r = solve_reference(d, scenarios, seconds, log_file=output/f'{name}-gurobi.log')
        save(output/f'{name}.json', dict(r, instance_files=instance_hashes(name)))
        print(f"REFERENCE {name} status={r['status']} objective={r['objective']} "
              f"gap={r.get('gurobi_gap_percent')} elapsed={r.get('elapsed_seconds')}", flush=True)
        pilot.ruler(r)
    return output


# 读取参考解目录中的 <算例>.json，核对它求解时的实例文件与当前一致，返回可当尺子的精简记录与元数据。
def load_reference(name, reference_dir):
    path = Path(reference_dir)/f'{name}.json'
    r = read(path)
    if r.get('instance_files') != instance_hashes(name):
        raise RuntimeError(f'{name}: reference was solved on different instance files')
    meta = dict(source=path.relative_to(ROOT).as_posix(), status=r['status'], objective=r['objective'],
                gurobi_gap_percent=r.get('gurobi_gap_percent'), gurobi_bound=r.get('gurobi_bound'),
                elapsed_seconds=r.get('elapsed_seconds'))
    return pilot.ruler(r), meta


# 冻结时记录的参考解文件若已不在原处（RTS24 曾缓存在已删除的筛选目录），改用 ieee118s10 参考解目录中的同名副本。
def reference_file(manifest, name):
    recorded = ROOT/manifest['references'][name]['source']
    return recorded if recorded.exists() else (ROOT/manifest['references']['ieee118s10']['source']).parent/f'{name}.json'


def freeze(output, smoke, references):
    files = [ROOT/n for n in SOURCES] + sorted((ROOT/'suc_benders').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    for name in COMPARE:
        hashes.update(instance_hashes(name))
    save(output/'manifest.json', dict(created_at=datetime.now().astimezone().isoformat(),
         git_head=git('rev-parse', 'HEAD').strip(), git_status=git('status', '--short'), python=sys.version,
         files=hashes, references=references))
    save(output/'protocol.json', dict(
        smoke=smoke, instances=COMPARE, seeds=SEEDS, seconds=SECONDS, workers=WORKERS, arms=pilot.ARMS,
        settings='cold start, run_suc-equivalent initialization, block_hours=12, sa_sweeps=16, LLM off',
        arms_meaning=dict(A='SA+Benders without LLM (S1)', **{'L+SA': 'SA+Benders plus the frozen offline '
                          'program programs/L3.py (S3); zero online LLM calls'}),
        l3_context='probability-weighted load and dispatch (identical to the old context for one scenario); '
                   'feasibility still checked on every scenario by necessary() and exact full-day dispatch',
        qubo_sizes=f'per master solve: ours = q.Q.shape[0]; standard lower bounds = free + {pilot.THETA_BITS} theta '
                   f'bits + (1 or {pilot.THETA_BITS}) bits per cut, for the same stored cuts and for single-cut '
                   'Benders (one cut per master solve)',
        criteria=CRITERIA, decisions={f'M1={k[0]},M2={k[1]}': v for k, v in DECISIONS.items()},
        case=f'start from ieee118s10 seed 17 A final schedule; call L3 at most {pilot.CASE_CALLS} times with the '
             'incumbent held; take the first improving candidate in proposal order; if none, next seed in '
             f'{SEEDS}, then rts24',
        reference_usage='gap ruler only; never injected as a candidate'))


def run_all(output, jobs, refs, seconds, workers):
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(pilot.counted_trial, name, arm, seed, seconds, refs[name]): (name, arm, seed)
                   for name, arm, seed in jobs}
        for future in as_completed(futures):
            row = future.result()
            save(output/'runs'/f"{row['instance']}-s{row['seed']}-{row['arm']}.json", row)
            print(f"RUN {row['instance']}-s{row['seed']} {row['arm']} gap={row['gap_percent']} "
                  f"status={row['status']} sa_calls={row['sa_calls']} operator_calls={row['operator_calls']}", flush=True)


# 每个（算例，种子）配对内轮换两臂的提交顺序。
def jobs_for(names, seeds):
    jobs = []
    for j, (name, seed) in enumerate((n, s) for n in names for s in seeds):
        arms = pilot.ARMS if j % 2 == 0 else pilot.ARMS[::-1]
        jobs += [(name, arm, seed) for arm in arms]
    return jobs


def summarize(runs):
    criteria = {key: pilot.evaluate(runs, key.split()[1], SEEDS) for key in CRITERIA}
    m1, m2 = (criteria[k]['passed'] for k in CRITERIA)
    return dict(criteria=criteria, verdict=DECISIONS[(m1, m2)])


def fmt(value, digits=3, suffix=''):
    return '—' if value is None else f'{value:.{digits}f}{suffix}'


def table_rows(runs, names, seeds):
    rows = []
    for name in names:
        for seed in seeds:
            pair = {r['arm']: r for r in runs if r['instance'] == name and r['seed'] == seed}
            if set(pilot.ARMS) <= set(pair):
                a, l = pair['A'], pair['L+SA']
                rows.append((name, seed, a, l, pilot.time_to_reach(l['trajectory'], a['objective'])))
    return rows


def report(output, smoke=False):
    from paper_figures import bits_figure, convergence_figure
    runs = [read(p) for p in sorted((output/'runs').glob('*.json'))]
    manifest, protocol = read(output/'manifest.json'), read(output/'protocol.json')
    names = [n for n in COMPARE if any(r['instance'] == n for r in runs)]
    seeds = sorted({r['seed'] for r in runs})
    summary = summarize(runs)
    seconds = runs[0]['seconds'] if runs else SECONDS
    lines = ['# 论文补充实验：SA+Benders 不加（A）与加离线程序 L3（L+SA）', '',
             f"每次 {seconds:g} 秒冷启动；在线 LLM 调用为 0；参考解只当差距尺子。"
             + ('**冒烟运行，判据无意义。**' if smoke else ''), '',
             '## T1 算例与参考解', '', '| 算例 | 机组×小时×场景 | 参考成本 | 状态 | Gurobi gap | 求解用时 (s) | 来源 |',
             '|---|---|---:|---|---:|---:|---|']
    for name in names:
        d, scenarios = pilot.instance(name)
        ref = manifest['references'][name]
        lines.append(f"| {name} | {d.n_units}×{d.horizon}×{len(scenarios)} | {ref['objective']:.2f} | {ref['status']} | "
                     f"{fmt(ref['gurobi_gap_percent'], 4, '%')} | {fmt(ref['elapsed_seconds'], 1)} | "
                     f"{reference_file(manifest, name).relative_to(ROOT).as_posix()} |")
    lines += ['', '## T2 终点差距与 AUC', '',
              '| 算例 | 种子 | A 差距 | L+SA 差距 | A AUC | L+SA AUC | AUC 降幅 | L+SA 达到 A 终点 (s) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, seed, a, l, reach in table_rows(runs, names, seeds):
        reduction = 1 - l['auc']/a['auc'] if a['auc'] else None
        lines.append(f"| {name} | {seed} | {fmt(a['gap_percent'], 3, '%')} | {fmt(l['gap_percent'], 3, '%')} | "
                     f"{fmt(a['auc'], 3)} | {fmt(l['auc'], 3)} | {fmt(None if reduction is None else 100*reduction, 1, '%')} | "
                     f"{fmt(reach, 1)} |")
    lines += ['', '## 来源拆分与 L3 候选去向', '',
              '| 运行 | 状态 | SA 降幅 | 程序降幅 | SA 调用 | 程序调用 | SA 用时 (s) | 程序用时 (s) | 候选去向 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(runs, key=lambda r: (r['instance'], r['seed'], r['arm'])):
        outcomes = ', '.join(f'{k}={v}' for k, v in sorted(r['operator_outcomes'].items())) or '—'
        lines.append(f"| {r['instance']}-s{r['seed']} {r['arm']} | {r['status']} | {r['decrease']['sa']:.0f} | "
                     f"{r['decrease']['operator']:.0f} | {r['sa_calls']} | {r['operator_calls']} | {r['sa_seconds']:.1f} | "
                     f"{r['operator_seconds']:.1f} | {outcomes} |")
    lines += ['', '## E2 QUBO 位数（每块最后一次主问题求解）', '',
              '标准编码为下界：θ 取 12 位，每条割的松弛寄存器取 1 位（宽松）或 12 位（与 θ 同分辨率）。'
              '"同一批割"指编码本文同一批已存割；"单割 Benders"指每次主问题求解只多一条割。'
              '下界只计自由位、θ 与割的松弛位，不计两种编码共有的容量行与边界行松弛位，因此偏小。', '',
              '| 运行 | 块 | 主问题次数 | 自由位 | 本文 | 已存最优性割 | 已存可行割 | 编码可行割 | 同一批割（1 位） | '
              '同一批割（12 位） | 单割 Benders（12 位） |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(runs, key=lambda r: (r['instance'], r['seed'], r['arm'])):
        for block in sorted({s['block'] for s in r['qubo_sizes']}):
            s = [x for x in r['qubo_sizes'] if x['block'] == block][-1]
            loose, fine = pilot.standard_bits(s, 1), pilot.standard_bits(s, pilot.THETA_BITS)
            lines.append(f"| {r['instance']}-s{r['seed']} {r['arm']} | {block} | {s['master']} | {s['free']} | {s['ours']} | "
                         f"{s['stored_optimality']} | {s['stored_feasibility']} | {s['feasibility_encoded']} | "
                         f"{loose['same_cuts']} | {fine['same_cuts']} | {fine['single_cut']} |")
    archive = read(ONLINE)
    found = {Path(e['directory']).name: e for e in archive['experiments']}
    lines += ['', '## T3 在线 LLM 的已有结果（' + ONLINE.relative_to(ROOT).as_posix() + '，单种子、旧版流程）', '',
              '| 算例 | 预算 (s) | 机组×小时×场景 | 纯 SA 差距 | SA+在线 LLM 差距 | 模型 | LLM 调用 | 状态 |',
              '|---|---:|---|---:|---:|---|---:|---|']
    for label, plain, online in ONLINE_PAIRS:
        a, l = found[plain], found[online]
        lines.append(f"| {label} | {a['time_limit']:g} | {a['units']}×{a['hours']}×{a['scenarios']} | "
                     f"{a['reference_gap_percent']:.3f}% | {l['reference_gap_percent']:.3f}% | {l['llm_model']} | "
                     f"{l['llm_calls']} | {l['status']} |")
    lines += ['', f"单次调用耗时：{archive['findings']['gpt_latency']}；{archive['findings']['deepseek_latency']}。"]
    lines += ['', '## 预先登记的判据', '']
    for key, value in summary['criteria'].items():
        lines.append(f"- **{key}**：{'通过' if value['passed'] else '未通过'}（完整配对 {value['pairs']}，胜 {value['wins']}，"
                     f"AUC 降幅中位数 {fmt(None if value['auc_reduction'] is None else 100*value['auc_reduction'], 1, '%')}）；"
                     f"标准：{CRITERIA[key]}")
    lines += ['', f"**判定**：{summary['verdict']}"]
    figures = output/'figures'
    convergence_figure(runs, names, figures/'F1-convergence')
    sized = [r for r in runs if r['arm'] == 'L+SA' and r['seed'] == min(seeds) and r['qubo_sizes']]
    bits_figure(sized, figures/'F2-qubo-bits')
    lines += ['', '## 图', '', '- F1 收敛曲线：`figures/F1-convergence.png`',
              '- F2 QUBO 位数：`figures/F2-qubo-bits.png`', f"- 协议：`protocol.json`（{len(protocol['seeds'])} 个种子）"]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(output/'summary.json', summary)
    return summary


# 案例：按预先登记的顺序找第一个由 L3 改进停滞方案的例子。
def run_case(compare_dir):
    from paper_figures import case_figure
    output = stamp('case')
    manifest = read(compare_dir/'manifest.json')
    order = [('ieee118s10', s) for s in SEEDS] + [('rts24', s) for s in SEEDS]
    tried = []
    for name, seed in order:
        path = compare_dir/'runs'/f'{name}-s{seed}-A.json'
        if not path.exists():
            continue
        start = read(path)
        ref = read(reference_file(manifest, name))
        row = pilot.case_study(name, seed, start['final_u'], pilot.ruler(ref))
        tried.append(dict(instance=name, seed=seed, calls=row['calls'], found=row['found']))
        if row['found']:
            row['tried'] = tried
            save(output/'case.json', row)
            case_figure(row, pilot.instance(name)[0], output/'F3-case')
            print(f"CASE {name}-s{seed} delta={row['delta']:.2f} units={row['changed_units']} calls={row['calls']}")
            return output
    save(output/'case.json', dict(found=False, tried=tried))
    print('CASE none found')
    return output


def audit(output):
    runs = [read(p) for p in sorted((output/'runs').glob('*.json'))]
    stored, manifest = read(output/'summary.json'), read(output/'manifest.json')
    recomputed = summarize(runs)
    checks = []
    for name in COMPARE:
        rows = sorted((r for r in runs if r['instance'] == name), key=lambda r: r['gap_percent'])[:2]
        if not rows:
            continue
        d, scenarios = pilot.instance(name)
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        for r in rows:
            result = dispatch(full, pack(d, np.asarray(r['final_u'], float)), 300.)
            plan = unpack(m, result.point) if result.feasible else None
            ok = bool(plan is not None and validate(d, scenarios, plan)['feasible'])
            checks.append(dict(run=f"{name}-s{r['seed']}-{r['arm']}", recorded=r['objective'], redispatched=result.cost,
                               feasible=ok, match=bool(abs(result.cost-r['objective']) <= 1e-6*max(1., abs(r['objective'])))))
    hashes = {name: {k: v for k, v in manifest['files'].items() if k in instance_hashes(name)} == instance_hashes(name)
              for name in COMPARE}
    references = {name: bool(load_reference(name, reference_file(manifest, name).parent)[1]['objective'] == ref['objective'])
                  for name, ref in manifest['references'].items()}
    sizes = all(s['ours'] >= s['free'] for r in runs for s in r['qubo_sizes'])
    value = dict(verdict_match=recomputed['verdict'] == stored['verdict'],
                 criteria_match=all(recomputed['criteria'][k]['passed'] == stored['criteria'][k]['passed'] for k in CRITERIA),
                 instance_hashes=hashes, references=references, qubo_sizes_consistent=sizes, solution_checks=checks)
    value['passed'] = bool(value['verdict_match'] and value['criteria_match'] and all(hashes.values())
                           and all(references.values()) and sizes and all(c['feasible'] and c['match'] for c in checks))
    save(output/'audit.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', nargs='+', metavar='INSTANCE', help='单独求解参考解，缓存到同一目录')
    parser.add_argument('--seconds', type=float, default=REFERENCE_SECONDS, help='参考解时间上限')
    parser.add_argument('--smoke', type=Path, metavar='REFERENCE_DIR', help='ieee118s10 seed 17，A 与 L+SA 各 90 秒')
    parser.add_argument('--compare', type=Path, metavar='REFERENCE_DIR', help='主实验；参数为含 ieee118s10、rts24 参考解的目录')
    parser.add_argument('--case', type=Path, metavar='COMPARE_DIR', help='案例')
    parser.add_argument('--report', type=Path, metavar='DIR', help='重新生成报告与图')
    parser.add_argument('--audit', type=Path, metavar='DIR', help='复核已完成的对比目录')
    args = parser.parse_args()
    if args.reference:
        print(run_reference(args.reference, args.seconds))
    elif args.audit:
        value = audit(args.audit.resolve())
        print(json.dumps(value, ensure_ascii=False, indent=1))
        return 0 if value['passed'] else 1
    elif args.report:
        print('VERDICT '+report(args.report.resolve(), read(args.report/'protocol.json')['smoke'])['verdict'])
    elif args.case:
        print(run_case(args.case.resolve()))
    elif args.smoke:
        output = stamp('smoke')
        output.mkdir(parents=True)
        print('OUTPUT '+str(output), flush=True)
        ruler, meta = load_reference('ieee118s10', args.smoke.resolve())
        freeze(output, True, dict(ieee118s10=meta))
        run_all(output, jobs_for(('ieee118s10',), (17,)), dict(ieee118s10=ruler), SMOKE_SECONDS, 2)
        report(output, smoke=True)
    elif args.compare:
        output = stamp('compare')
        output.mkdir(parents=True)
        print('OUTPUT '+str(output), flush=True)
        loaded = {name: load_reference(name, args.compare.resolve()) for name in COMPARE}
        freeze(output, False, {name: meta for name, (_, meta) in loaded.items()})
        run_tests(output)
        run_all(output, jobs_for(COMPARE, SEEDS), {name: ruler for name, (ruler, _) in loaded.items()}, SECONDS, WORKERS)
        print('VERDICT '+report(output)['verdict'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
