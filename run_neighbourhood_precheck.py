"""阶段 0 预检（不用 LLM）：在 v1.2.0 中 SA+L24 的停滞点上，SA+Benders 只在给定邻域内搜索，能否找到改进。

用法：py -3.10 run_neighbourhood_precheck.py <experiment-archive-v1.2.0.zip> [--smoke]
预登记判据见 protocol.json：N_rule 上 SA+Benders 30 秒内在 >= 3/4 个停滞点找到严格改进为 PASS，否则 STOP。
N_oracle（用参考解）、RM（邻域内随机改动，不用 SA）、C_N 与至多改 k 台的最优只作描述，Gurobi 只当尺子。
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

import neighbourhood_search as ns
from suc_benders.config import Options
from suc_benders.data import jsonable, load_data, save_json

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = '7f5bd7647ee1130ac636fd83b17fbfef2f847c4fd640687aca681dbab6d7fb68'
EVALUATION = 'experiment-archive-v1.2.0/results/lit-evaluation-rts24-20261026-213117-7bb610af'
# 停滞点：v1.2.0 验证中 SA+L24（归档名 LIT+SA）的终点方案。
POINTS = [('rts24-minus', 83), ('rts24-minus', 101), ('rts24-base', 17), ('rts24-plus', 83)]
SECONDS = dict(sa=30., rm=30., ceiling=60., budget=90.)
BUDGET_K = (4, 6, 8)
GATE = dict(neighbourhoods='N_rule', method='SA+Benders', seconds=30., points=4, pass_points=3,
            rel_tolerance=ns.REL_TOLERANCE)
CODE = ['neighbourhood_search.py', 'run_neighbourhood_precheck.py', 'offline_pilot.py', 'offline_worker.py',
        'requirements.txt', 'tests/test_neighbourhood_search.py', 'tests/conftest.py',
        *sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'suc_benders').glob('*.py'))]


def now():
    return datetime.now().astimezone().isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def options(seed, seconds):
    return Options(seed=seed, time_limit=seconds, block_hours=12, max_rounds=10**6, sa_reads=8, sa_sweeps=16,
                   sa_temperature=4., sa_cooling=.95, sa_cutoff=.05, llm_enabled=False, verbose=False)


def gap(objective, reference):
    return None if objective is None else 100*(objective-reference)/abs(reference)


# 从归档 zip 取停滞点、输入与参考解；输入写进输出目录并核对参考解登记的哈希。
def load_points(archive, out, points):
    data = archive.read_bytes()
    if sha256(data) != ARCHIVE_SHA256:
        raise RuntimeError('Archive SHA256 mismatch')
    rows = []
    with zipfile.ZipFile(archive) as z:
        for variant, seed in points:
            reference = json.loads(z.read(f'{EVALUATION}/references/{variant}.json'))
            if reference['status'] != 'OPTIMAL':
                raise RuntimeError(f'{variant}: reference not OPTIMAL')
            folder = out/'inputs'/variant
            folder.mkdir(parents=True, exist_ok=True)
            for name, expected in reference['instance_files'].items():
                raw = z.read(f'{EVALUATION}/inputs/{variant}/{name}')
                if sha256(raw) != expected:
                    raise RuntimeError(f'{variant}/{name}: hash mismatch')
                (folder/name).write_bytes(raw)
            run = json.loads(z.read(f'{EVALUATION}/runs/{variant}-s{seed}-LIT+SA.json'))
            rows.append(dict(variant=variant, seed=seed, folder=folder, run_objective=run['objective'],
                             run_gap=run['reference_gap_percent'], u=np.asarray(run['final_u'], float),
                             reference=dict(status=reference['status'], objective=reference['objective'],
                                            validation=reference['validation']),
                             reference_u=np.rint(np.abs(np.asarray(reference['u'], float)))))
    return rows


# 一次邻域内 SA+Benders：新建全天状态，按轮转逐个邻域 focus 再采样一批，直到 seconds 用完。
def sa_measure(point, d, scenarios, neighbourhoods, seconds):
    o = options(point['seed'], seconds)
    whole = ns.start_whole(d, scenarios, o, point['u'], point['reference'], seconds+600.)
    start_objective = whole.result.objective
    started = perf_counter()
    whole.started, whole.deadline = started, started+seconds
    search = ns.NeighbourhoodSearch(whole, o)
    steps, k = [], 0
    while whole.remaining() > 0:
        n = neighbourhoods[k % len(neighbourhoods)]; k += 1
        free = search.focus(n)
        row = search.step(k)
        if row is None:
            break
        steps.append(dict(row, free_after_focus=free))
    return summarize(whole, start_objective, point, dict(steps=steps, sa_calls=whole.diagnostics['sa_calls'],
                                                         cuts=search.cut_totals))


def rm_measure(point, d, scenarios, neighbourhoods, seconds):
    o = options(point['seed'], seconds)
    whole = ns.start_whole(d, scenarios, o, point['u'], point['reference'], seconds+600.)
    start_objective = whole.result.objective
    started = perf_counter()
    whole.started, whole.deadline = started, started+seconds
    counts = ns.random_moves(whole, neighbourhoods, seconds, point['seed'])
    return summarize(whole, start_objective, point, dict(counts=counts))


def summarize(whole, start_objective, point, extra):
    evaluations = [e for e in whole.diagnostics['evaluations'] if e['source'] != 'initial']
    first = next((e['elapsed_seconds'] for e in evaluations
                  if e['improved'] and ns.improves(e['objective'], start_objective)), None)
    objective = whole.result.objective
    return dict(start_objective=float(start_objective), objective=float(objective),
                gap_percent=gap(objective, point['reference']['objective']),
                improved=bool(ns.improves(objective, start_objective)), first_improvement_seconds=first,
                validated=bool((whole.result.validation or {}).get('feasible')), evaluations=len(evaluations),
                decrease=ns.decrease_by_source(whole.diagnostics['evaluations']),
                final_u=np.asarray(whole.result.u, int).tolist(), timings=dict(whole.timings), **extra)


def measure(point, seconds, budget_k, out):
    d, scenarios = load_data(point['folder']/'units.json', point['folder']/'stochastic.json')
    name = f"{point['variant']}-s{point['seed']}"
    # 停滞点身份：重新调度的已校验费用须与归档运行的终点一致。
    whole = ns.start_whole(d, scenarios, options(point['seed'], 60.), point['u'], point['reference'], 600.)
    identity = abs(whole.result.objective-point['run_objective']) <= 1e-6*abs(point['run_objective'])
    rule = ns.rule_neighbourhoods(d, whole.result)
    oracle = ns.oracle_neighbourhood(d, point['u'], point['reference_u'])
    m, full = whole.m, whole.full
    row = dict(point=name, variant=point['variant'], seed=point['seed'], incumbent=float(whole.result.objective),
               run_objective=point['run_objective'], identity=bool(identity),
               reference=point['reference']['objective'], incumbent_gap_percent=gap(whole.result.objective,
                                                                                   point['reference']['objective']),
               neighbourhoods=dict(rule=[ns.describe(n) for n in rule], oracle=ns.describe(oracle)))
    print(f'[{name}] incumbent gap {row["incumbent_gap_percent"]:.4f}% identity={identity}', flush=True)
    for label, group in (('rule', rule), ('oracle', [oracle])):
        row[f'sa_{label}'] = sa_measure(point, d, scenarios, group, seconds['sa'])
        print(f"[{name}] SA in N_{label}: improved={row[f'sa_{label}']['improved']} "
              f"gap {row[f'sa_{label}']['gap_percent']:.4f}% first={row[f'sa_{label}']['first_improvement_seconds']}",
              flush=True)
        row[f'rm_{label}'] = rm_measure(point, d, scenarios, group, seconds['rm'])
        print(f"[{name}] RM in N_{label}: improved={row[f'rm_{label}']['improved']} "
              f"gap {row[f'rm_{label}']['gap_percent']:.4f}%", flush=True)
        row[f'ceiling_{label}'] = []
        for n in group:
            c = ns.neighbourhood_ceiling(m, full, point['u'], n, seconds['ceiling'])
            c.update(neighbourhood=ns.describe(n), gap_percent=gap(c.get('objective'), row['reference']),
                     improves=bool(ns.improves(c.get('objective'), row['incumbent'])))
            row[f'ceiling_{label}'].append(c)
            print(f"[{name}] C_N {label}: {c['status']} gap {c['gap_percent']} changed_units={c.get('changed_units')}",
                  flush=True)
    row['unit_budget'] = {}
    for k in budget_k:
        c = ns.unit_budget(m, full, point['u'], k, seconds['budget'])
        c.update(gap_percent=gap(c.get('objective'), row['reference']),
                 improves=bool(ns.improves(c.get('objective'), row['incumbent'])))
        row['unit_budget'][k] = c
        print(f"[{name}] <= {k} units: {c['status']} gap {c['gap_percent']} bound {c.get('bound')}", flush=True)
    save_json(out/'runs'/f'{name}.json', jsonable(row))
    return row


def decide(rows):
    hits = sum(r['sa_rule']['improved'] and r['sa_rule']['validated'] for r in rows)
    valid = all(r['identity'] for r in rows) and len(rows) == GATE['points']
    decision = 'PASS' if valid and hits >= GATE['pass_points'] else ('STOP' if valid else 'INVALID')
    return dict(decision=decision, points_improved=hits, points=len(rows), identity=all(r['identity'] for r in rows))


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def report(out, rows, verdict):
    lines = ['# 阶段 0 预检：邻域内 SA+Benders', '', f'生成时间：{now()}', '',
             f"**判据结论：{verdict['decision']}**（N_rule 上 SA+Benders 30 秒内有严格改进的停滞点 "
             f"{verdict['points_improved']}/{verdict['points']}，门槛 {GATE['pass_points']}/{GATE['points']}；"
             f"停滞点身份核对 {'通过' if verdict['identity'] else '未通过'}）", '',
             '差距均相对参考最优（Gurobi，只当尺子），单位 %。', '',
             '| 停滞点 | 起点差距 | SA·N_rule | 首次改进 s | SA·N_oracle | RM·N_rule | RM·N_oracle '
             '| C_N rule 最好 | C_N oracle | ≤4 台 | ≤6 台 | ≤8 台 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def cell(c):
        return '—' if c is None else fmt(c.get('gap_percent')) + ('' if c.get('proven') else '*')

    for r in rows:
        best_rule = min((c for c in r['ceiling_rule'] if c.get('gap_percent') is not None),
                        key=lambda c: c['gap_percent'], default=None)
        budget = r['unit_budget']
        lines.append(f"| {r['point']} | {fmt(r['incumbent_gap_percent'])} | {fmt(r['sa_rule']['gap_percent'])} "
                     f"| {fmt(r['sa_rule']['first_improvement_seconds'], 1)} | {fmt(r['sa_oracle']['gap_percent'])} "
                     f"| {fmt(r['rm_rule']['gap_percent'])} | {fmt(r['rm_oracle']['gap_percent'])} | {cell(best_rule)} "
                     f"| {cell(r['ceiling_oracle'][0])} " + ' '.join(f'| {cell(budget.get(k))}' for k in BUDGET_K) + ' |')
    lines += ['', '\\* 未在时限内证明最优，取时限内最好可行解。', '',
              '- SA·N_rule / SA·N_oracle：邻域内 SA+Benders 30 秒后的差距；RM：同一邻域内随机改动（不用 SA）30 秒。',
              '- C_N：邻域内精确最优（邻域外 u 固定）；≤k 台：全天任意小时、至多改 k 台机组的最优。', '',
              '## 邻域', '']
    for r in rows:
        lines.append(f"- {r['point']}：N_rule = {json.dumps(r['neighbourhoods']['rule'], ensure_ascii=False)}；"
                     f"N_oracle 机组 = {r['neighbourhoods']['oracle']['units']}")
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out = ROOT/'results'/f"neighbourhood-precheck-{'smoke-' if args.smoke else ''}{stamp}"
    (out/'runs').mkdir(parents=True)
    points = POINTS[:1] if args.smoke else POINTS
    seconds = dict(sa=5., rm=5., ceiling=20., budget=20.) if args.smoke else SECONDS
    budget_k = (4,) if args.smoke else BUDGET_K
    snapshot = out/'snapshot'
    for name in CODE:
        (snapshot/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, snapshot/name)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rows = load_points(args.archive, out, points)
    save_json(out/'protocol.json', jsonable(dict(
        created_at=now(), smoke=args.smoke, question='Given a neighbourhood, can SA+Benders improve the SA+L24 stall points?',
        archive=dict(path=str(args.archive), sha256=ARCHIVE_SHA256, evaluation=EVALUATION),
        points=[f'{v}-s{s}' for v, s in points], seconds=seconds, unit_budget_k=list(budget_k),
        options={k: v for k, v in vars(options(0, 0.)).items() if not k.startswith('llm') and 'file' not in k},
        neighbourhoods=dict(rule='top-4 committed units by average cost per MWh (no-load + startup + expected fuel, '
                                 'divided by expected energy); window = on-hours +/- 1 h; '
                                 'companions = 11 smallest-pmax units off in some window hour',
                            rule_note='The smoke run before this protocol used a deviating target score (wasted '
                                      'commitment cost); restored to the pre-registered average cost per MWh before '
                                      'the formal run.',
                            oracle='all units of identical-unit groups whose hourly on-counts differ from the '
                                   'reference, all 24 hours (diagnostic only; uses the reference)'),
        gate=GATE, decisions=dict(PASS='SA+Benders in N_rule improves >= 3/4 stall points within 30 s',
                                  STOP='otherwise; no LLM training'),
        reference_usage='Gurobi references, C_N and unit-budget MIPs are rulers only')))
    save_json(out/'manifest.json', jsonable(dict(
        created_at=now(), git_head=git, python=sys.version, platform=platform.platform(),
        packages={p: version(p) for p in ('numpy', 'scipy', 'gurobipy', 'dwave-samplers', 'dimod')},
        snapshot={name: sha256((snapshot/name).read_bytes()) for name in CODE},
        inputs={str(p.relative_to(out)).replace('\\', '/'): sha256(p.read_bytes())
                for p in sorted((out/'inputs').rglob('*.json'))})))
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=ROOT,
                           capture_output=True, text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
    print(tests.stdout.strip().splitlines()[-1], flush=True)
    if tests.returncode != 0:
        raise SystemExit('Tests failed; see tests.log')
    started = perf_counter()
    results = [measure(p, seconds, budget_k, out) for p in rows]
    verdict = decide(results) if not args.smoke else dict(decision='SMOKE', points_improved=sum(
        r['sa_rule']['improved'] for r in results), points=len(results), identity=all(r['identity'] for r in results))
    save_json(out/'summary.json', jsonable(dict(updated_at=now(), verdict=verdict, seconds=perf_counter()-started,
                                                 points=[{k: r[k] for k in ('point', 'incumbent_gap_percent', 'identity')}
                                                         | {f'{m}_{l}_gap': r[f'{m}_{l}']['gap_percent']
                                                            for m in ('sa', 'rm') for l in ('rule', 'oracle')}
                                                         for r in results])))
    report(out, results, verdict)
    print(f"{verdict['decision']}: {verdict['points_improved']}/{verdict['points']} -> {out}", flush=True)


if __name__ == '__main__':
    main()
