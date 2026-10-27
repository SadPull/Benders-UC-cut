"""程序验证实验的协议、测试实例、计分与判据：rts24 原工况与两种负荷扰动，所有对照方法重新运行。

测试负荷扰动、参考可用性、计分与配对移植自已归档的 paper_supplement.py（见 LLM-benders-SUC-archive-20261026.zip）；
写出的测试实例与补充实验所用文件逐字节相同（TEST_FILES），新旧结果可以按实例对照。
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

import numpy as np

import offline_pilot
from run_paper_pilot import save
from suc_benders.data import load_data

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'instances/rts24'
CASE = 'rts24'
# 方法角色：无 LLM 对照、IEEE118 上生成的程序、早期程序、终选程序（见 offline_pilot.ARMS）。
CONTROL_ARM, L118_ARM, EARLY_ARM, FINAL_ARM = 'SA+R', 'SA+L118', 'SA+L24r2', 'SA+L24'
BASE_ARMS = (CONTROL_ARM, L118_ARM, EARLY_ARM, FINAL_ARM)
SHIFT_ARMS = (CONTROL_ARM, L118_ARM, FINAL_ARM)
SEEDS = (17, 29, 41, 53, 67)
SHIFT_SEEDS = (83, 101, 137)
# runs 在名义 17,400 秒之外为每次运行的进程启动与保存留约 15 秒。
LIMITS = dict(checks=300., references=900., runs=18000., audit=1200.)
# 补充实验 inputs/ 中的同名文件（2026-10-25）。
TEST_FILES = {
    'rts24-base/units.json': 'e841e29a32df13449a42f589c766e13adab80ca54a644781c25ea43f6c0162e6',
    'rts24-base/stochastic.json': 'ef1449667234504c3ced04a25522fbb9e4c6171268d12d9abbc6ca1392f0a0df',
    'rts24-minus/units.json': 'e841e29a32df13449a42f589c766e13adab80ca54a644781c25ea43f6c0162e6',
    'rts24-minus/stochastic.json': '60fc110b58119ae3b3f517163dacd74e51133e5b4ab72bd59f6f1ec553d6e7de',
    'rts24-plus/units.json': 'e841e29a32df13449a42f589c766e13adab80ca54a644781c25ea43f6c0162e6',
    'rts24-plus/stochastic.json': '5ee8712ad1e21bd38fdba0bb77746286aba31f478ae7fdf45bcc8c46649a2b30'}
# 判据门槛：终点胜出次数、相对 AUC 降幅中位数。
WINS, AUC_REDUCTION, L118_WINS = 4, .10, 8
CRITERIA = dict(
    T1='Iteration effect (primary): rts24-base, SA+L24 vs SA+L24r2, 5 complete pairs, >=4 final-cost wins and '
       'median AUC difference >0; NO_IMPROVEMENT_AFTER_ROUND_1 when the final program is the early program',
    T2='Versus no LLM: rts24-base, SA+L24 vs SA+R, 5 complete pairs, >=4 final-cost wins and median relative AUC reduction >=10%',
    T3='Versus L118: rts24 base+minus+plus (11 pairs), SA+L24 vs SA+L118, >=8 final-cost wins and median AUC difference >0',
    descriptive='SA+L118 and SA+L24r2 vs SA+R; final reference gaps',
    note='Engineering gates, not significance tests')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturb(source, sign):
    value = deepcopy(source)
    for scenario in value['scenarios']:
        old = np.asarray(scenario['load'], float)
        if len(old) != 24 or sign not in (-1, 1):
            raise ValueError('The frozen perturbation requires 24 hours and sign -1 or 1')
        x = np.arange(24)/23
        load = np.round(old*(1+sign*.015*np.sin(np.pi*x)+.01*np.sin(4*np.pi*x)), 2)
        load[[0, -1]] = old[[0, -1]]
        scenario['load'] = load.tolist()
    return value


# 与补充实验 prepare() 相同：原工况直接复制，扰动工况先把场景负荷写成显式数组。
def write_instances(folder, source=SOURCE):
    d, scenarios = load_data(source/'units.json', source/'stochastic.json')
    explicit = dict(reserve_fraction=d.reserve_fraction, scenarios=[dict(
        name=s.name, probability=s.probability, load=s.load.tolist()) for s in scenarios])
    for v, sign in (('base', 0), ('minus', -1), ('plus', 1)):
        target = folder/f'{CASE}-{v}'
        target.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source/'units.json', target/'units.json')
        if sign:
            save(target/'stochastic.json', perturb(explicit, sign))
        else:
            shutil.copyfile(source/'stochastic.json', target/'stochastic.json')
    changed = [name for name, h in TEST_FILES.items() if digest(folder/name) != h]
    if changed:
        raise ValueError('Test instances differ from the supplement files: '+', '.join(changed))
    return sorted(p.name for p in folder.iterdir())


def instances():
    return [f'{CASE}-{v}' for v in ('base', 'minus', 'plus')]


# 每个配对内各方法的运行顺序按配对序号轮换。include_early=False 时（终选程序就是早期程序）不跑早期程序。
def jobs(include_early=True):
    rows, pair = [], 0
    for variant, seeds, seconds, arms in (('base', SEEDS, 600., BASE_ARMS),
                                          ('minus', SHIFT_SEEDS, 300., SHIFT_ARMS),
                                          ('plus', SHIFT_SEEDS, 300., SHIFT_ARMS)):
        arms = [a for a in arms if include_early or a != EARLY_ARM]
        for seed in seeds:
            k = pair % len(arms)
            for arm in arms[k:]+arms[:k]:
                rows.append(dict(id=f'{CASE}-{variant}-s{seed}-{arm}', stage='runs', case=CASE, variant=variant,
                                 instance=f'{CASE}-{variant}', arm=arm, seed=seed, seconds=seconds))
            pair += 1
    return rows


def usable_reference(reference):
    cost, gap = reference.get('objective'), reference.get('gurobi_gap_percent')
    return bool(cost is not None and np.isfinite(cost) and (reference.get('validation') or {}).get('feasible')
                and (reference.get('status') == 'OPTIMAL' or
                     (reference.get('status') == 'TIME_LIMIT' and gap is not None and 0 <= gap <= .05)))


def score(run, reference):
    initial, curve = run.get('initial', {}), run.get('trajectory', [])
    c0 = initial.get('objective')
    valid = usable_reference(reference)
    cost = float(reference['objective']) if valid else None
    objective = run.get('objective')
    eps = 1e-6*max(1., abs(c0)) if c0 is not None else None
    # 初始化本身不算搜索改善；名义预算包含五秒收尾，末值一直延伸到 T。
    scoring_curve = ([(0., c0)] + [(t, c) for t, c in curve if t >= initial['seconds']]) if initial else []
    return dict(reference_cost=cost, reference_status=reference.get('status', 'UNAVAILABLE'),
                reference_proven=valid and reference.get('status') == 'OPTIMAL',
                reference_gurobi_gap_percent=reference.get('gurobi_gap_percent'),
                reference_gap_percent=100*(objective-cost)/abs(cost)
                    if cost not in (None, 0.) and objective is not None and run.get('feasible') else None,
                auc=offline_pilot.auc(scoring_curve, cost, run['seconds']),
                initial_u_sha256=hashlib.sha256(np.asarray(initial['u'], np.uint8).tobytes()).hexdigest() if initial else None,
                cost_tolerance=eps, first_improvement_seconds=next(
                    (t for t, c in curve if t >= initial['seconds'] and c < c0-eps), None) if initial else None)


# control 对 treatment：win 指 treatment 终点更低；差值为 control 减 treatment，正数有利于 treatment。
def pair_result(control, treatment):
    valid = all(r.get('status') == 'TIME_LIMIT' and r.get('feasible') and r.get('initial')
                and not r.get('over_budget', False) for r in (control, treatment))
    if not valid:
        return dict(valid=False, reason='RUN_INCOMPLETE')
    eps = control['cost_tolerance']
    same = (control['initial_u_sha256'] == treatment['initial_u_sha256']
            and abs(control['initial']['objective']-treatment['initial']['objective']) <= eps)
    if not same:
        return dict(valid=False, reason='INITIALIZATION_MISMATCH')
    a, b = control.get('auc'), treatment.get('auc')
    return dict(valid=True, win=treatment['objective'] < control['objective']-eps,
                cost_difference=control['objective']-treatment['objective'],
                auc_difference=a-b if a is not None and b is not None else None,
                auc_reduction=1-b/a if a is not None and a > 1e-9 and b is not None else None)


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize(runs, include_early=True, t1_reason='NO_IMPROVEMENT_AFTER_ROUND_1'):
    by = {(r['instance'], r['seed'], r['arm']): r for r in runs}
    keys = list(dict.fromkeys((j['instance'], j['seed']) for j in jobs(include_early)))

    def compare(control, treatment, variants):
        rows = []
        for instance, seed in keys:
            if instance.split('-')[1] not in variants:
                continue
            a, b = by.get((instance, seed, control)), by.get((instance, seed, treatment))
            outcome = pair_result(a, b) if a and b else dict(valid=False, reason='MISSING_RUN')
            rows.append(dict(instance=instance, seed=seed, control=control, treatment=treatment, **outcome))
        return dict(complete=all(p['valid'] for p in rows), pairs=len(rows), wins=sum(p.get('win', False) for p in rows),
                    median_auc_reduction=_median([p.get('auc_reduction') for p in rows]),
                    median_auc_difference=_median([p.get('auc_difference') for p in rows]),
                    auc_pairs=sum(p.get('auc_difference') is not None for p in rows), details=rows)

    if include_early:
        t1 = compare(EARLY_ARM, FINAL_ARM, ('base',))
        t1['status'] = ('INCOMPLETE' if not t1['complete'] or t1['auc_pairs'] != 5 else 'ITERATION_HELPS'
                        if t1['wins'] >= WINS and t1['median_auc_difference'] > 1e-9 else 'NOT_SUPPORTED')
    else:
        t1 = dict(status=t1_reason)
    t2 = compare(CONTROL_ARM, FINAL_ARM, ('base',))
    t2['status'] = ('INCOMPLETE' if not t2['complete'] or t2['auc_pairs'] != 5 else 'PASS'
                    if t2['wins'] >= WINS and t2['median_auc_reduction'] >= AUC_REDUCTION else 'FAIL')
    t3 = compare(L118_ARM, FINAL_ARM, ('base', 'minus', 'plus'))
    t3['status'] = ('INCOMPLETE' if not t3['complete'] else 'SUPPORTED'
                    if t3['wins'] >= L118_WINS and (t3['median_auc_difference'] or 0.) > 1e-9 else 'NOT_SUPPORTED')
    every = ('base', 'minus', 'plus')
    descriptive = dict(L118_vs_R_base=compare(CONTROL_ARM, L118_ARM, ('base',)), L24_vs_R_all=compare(CONTROL_ARM, FINAL_ARM, every),
                       L118_vs_R_all=compare(CONTROL_ARM, L118_ARM, every))
    if include_early:
        descriptive['L24r2_vs_R_base'] = compare(CONTROL_ARM, EARLY_ARM, ('base',))
    return dict(T1_iteration=t1, T2_vs_R=t2, T3_vs_L118=t3, descriptive=descriptive)
