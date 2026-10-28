"""v1.2.0 程序训练与验证实验的中文静态图：学习曲线、收敛曲线、按判据配对的 AUC 与终点差距，另附图例说明与数据表。
用法：py -3.10 program_figures.py <训练目录> <验证目录> <输出目录>
归档目录在统一命名之前生成，运行记录里的旧方法名按 LEGACY 映射为新名；数值不改动。
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import program_evaluation as study
from paper_figures import BLUE, INK, METHOD_STYLE, MUTED, ORANGE, gap_curve, save, style

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False})
LEGACY = {'A': 'SA', 'N+SA': 'SA+R', 'L+SA': 'SA+L118', 'LIT+SA': 'SA+L24', 'LIT1+SA': 'SA+L24r2'}
LABELS = {'SA': 'SA：只有 SA+Benders',
          'SA+R': 'SA+R：加随机算子 R（不用 LLM）',
          'SA+L118': 'SA+L118：加程序 L118（在 IEEE118 上生成）',
          'SA+L24r2': 'SA+L24r2：加程序 L24r2（RTS24 训练第 2 轮）',
          'SA+L24': 'SA+L24：加程序 L24（RTS24 训练 10 轮的终选）'}
VARIANT_NAMES = dict(base='原工况', minus='负荷下调', plus='负荷上调')
VARIANTS = (('base', '原工况（rts24-base）：5 个种子，各 600 秒'), ('minus', '负荷下调（rts24-minus）：3 个种子，各 300 秒'),
            ('plus', '负荷上调（rts24-plus）：3 个种子，各 300 秒'))
GREY_POINT, CONNECTOR = '#b9b8b3', '#d2d1cc'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_runs(evaluation_dir):
    protocol = read(evaluation_dir/'protocol.json')
    rows = [read(evaluation_dir/'runs'/f"{j['id']}.json") for j in protocol['jobs']]
    return [dict(r, arm=LEGACY.get(r['arm'], r['arm'])) for r in rows]


def scatter(ax, arm, x, y, size=46):
    s = METHOD_STYLE[arm]
    ax.scatter(x, y, marker=s['marker'], s=size, linewidths=1.4 if not s['filled'] else .8,
               facecolors=s['color'] if s['filled'] else 'white', edgecolors=s['color'] if not s['filled'] else 'white',
               zorder=4)


def marker_handle(arm):
    s = METHOD_STYLE[arm]
    return plt.Line2D([], [], linestyle='none', marker=s['marker'], markersize=7, markeredgewidth=1.4,
                      markerfacecolor=s['color'] if s['filled'] else 'white', markeredgecolor=s['color'])


def note(fig, text, y):
    fig.text(.01, y, text, ha='left', va='top', fontsize=7.5, color=MUTED)


# 学习曲线：每轮两个程序的验证均分（灰点）、截至该轮最佳 B_r（蓝色阶梯，实心为“改进”轮）、随机算子 R。
def learning_figure(curve, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rounds = [r['round'] for r in curve['rounds']]
    for r in curve['rounds']:
        values = [p['validation'] for p in r['programs'].values() if p['validation'] is not None]
        ax.scatter([r['round']]*len(values), values, s=30, color=GREY_POINT, zorder=2)
    best = [(r['round'], r['best_validation'], r['improved']) for r in curve['rounds'] if r['best_validation'] is not None]
    ax.step([b[0] for b in best], [b[1] for b in best], where='post', color=BLUE, linewidth=2, zorder=3)
    ax.scatter([b[0] for b in best if b[2]], [b[1] for b in best if b[2]], s=46, color=BLUE, edgecolors='white',
               linewidths=.8, zorder=4)
    ax.scatter([b[0] for b in best if not b[2]], [b[1] for b in best if not b[2]], s=46, facecolors='white',
               edgecolors=BLUE, linewidths=1.4, zorder=4)
    ax.axhline(curve['R_validation_mean'], color=ORANGE, linewidth=1.5, zorder=1)
    picks = {'P2a': 'L24r2 = P2a（早期对照）', 'P9b': 'L24 = P9b（终选）'}
    for r in curve['rounds']:
        for identifier, p in r['programs'].items():
            if identifier in picks:
                ax.annotate(picks[identifier], (r['round'], p['validation']), xytext=(0, 9), textcoords='offset points',
                            ha='center', fontsize=8, color=INK)
    ax.annotate('第 1 轮：两个程序都\n未通过静态检查，没有分数', (.65, .95), ha='left', va='center', fontsize=7.5, color=MUTED)
    ax.set_xticks(rounds); ax.set_xlim(.5, max(rounds)+.5); ax.set_ylim(0, 3.5)
    ax.set_xlabel('训练轮次', fontsize=8.5); ax.set_ylabel('验证均分 J（百分点，越高越好）', fontsize=8.5)
    ax.set_title(f"训练学习曲线：共 {curve['stop']['rounds']} 轮，达到轮数上限后停止", fontsize=9.5, color=INK)
    style(ax)
    handles = [plt.Line2D([], [], linestyle='none', marker='o', markersize=6, markerfacecolor=GREY_POINT, markeredgecolor=GREY_POINT),
               plt.Line2D([], [], color=BLUE, linewidth=2, marker='o', markersize=7, markerfacecolor=BLUE, markeredgecolor='white'),
               plt.Line2D([], [], color=BLUE, linewidth=2, marker='o', markersize=7, markerfacecolor='white', markeredgecolor=BLUE,
                          markeredgewidth=1.4),
               plt.Line2D([], [], color=ORANGE, linewidth=1.5)]
    labels = ['灰点：本轮生成的单个程序的验证均分', '蓝线实心点：截至本轮的最佳均分 B_r，本轮计为“改进”',
              '蓝线空心点：截至本轮的最佳均分 B_r，本轮未达到改进门槛', f"橙线：随机算子 R 的验证均分（{curve['R_validation_mean']:.2f}）"]
    ax.legend(handles, labels, frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0, .98))
    fig.tight_layout(rect=(0, .1, 1, 1))
    note(fig, 'J：从固定锚点出发做 15 秒探测，取前 10 秒内成本改善的时间平均（百分点）；每个程序取 2 个验证变体 × 3 个锚点的平均。\n'
              '“改进”门槛：B_r 至少比上一轮高 max(0.10, 3%×上一轮)。规则为至少 4 轮、连续 2 轮无改进即停、最多 10 轮。', .075)
    save(fig, path)


# 收敛曲线：每个工况一幅，粗线为各方法跨种子的中位数，细线为单个种子；时间取对数轴，改善大多发生在前 20 秒。
# SA+L24 的实线画在底层，与其他方法终点相同时（如负荷上调的 0.704%）虚线仍可见；初始化完成前没有可行解，曲线为空。
def convergence_figure(rows, path):
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(11.6, 3.9), squeeze=False)
    for ax, (variant, title) in zip(axes[0], VARIANTS):
        mine = [r for r in rows if r['variant'] == variant]
        grid = np.geomspace(.5, max(r['seconds'] for r in mine)-5, 800)
        for arm in study.BASE_ARMS:
            curves = np.array([gap_curve(r, grid) for r in mine if r['arm'] == arm])
            if not len(curves):
                continue
            s, z = METHOD_STYLE[arm], 2 if arm == study.FINAL_ARM else 3
            for c in curves:
                ax.step(grid, c, where='post', color=s['color'], linestyle=s['linestyle'], linewidth=.6, alpha=.3, zorder=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median = np.nanmedian(curves, axis=0)
            ax.step(grid, median, where='post', color=s['color'], linestyle=s['linestyle'], linewidth=2, zorder=z)
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, grid[-1]]); ax.set_xticklabels(['1', '10', '100', f'{grid[-1]+5:.0f}'])
        ax.set_ylim(0, 6); ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('计算时间（秒，对数刻度）', fontsize=8.5)
        style(ax)
    axes[0][0].set_ylabel('与参考最优解的差距（%，越低越好）', fontsize=8.5)
    handles = [plt.Line2D([], [], color=METHOD_STYLE[a]['color'], linestyle=METHOD_STYLE[a]['linestyle'], linewidth=2)
               for a in study.BASE_ARMS]
    handles += [plt.Line2D([], [], color=MUTED, linewidth=2), plt.Line2D([], [], color=MUTED, linewidth=.6, alpha=.6)]
    labels = [LABELS[a] for a in study.BASE_ARMS]+['粗线：同一方法各种子的中位数', '细线：单个种子']
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .07, 1, .86))
    note(fig, '所有方法在同一工况、同一种子下从同一个初始方案出发；约 1 秒的初始化完成前还没有可行解，因此曲线从约 0.8 秒开始。'
              '负荷上调工况中 SA+L118 有两个种子的终点与 SA+L24 相同（0.704%），两条线在后段重合。', .055)
    save(fig, path)


def outcome(control, treatment):
    result = study.pair_result(control, treatment)
    eps = control['cost_tolerance']
    return '胜' if result['win'] else '平' if abs(result['cost_difference']) <= eps else '负'


# 按判据配对的 AUC 哑铃图：每行一个（工况，种子）配对，两点分别是对照方法与 SA+L24；右侧注明 SA+L24 终点成本的胜负。
# 三幅共用横轴，判据之间的差距大小可以直接比较。
def gates_figure(rows, summary, path):
    verdicts = {'ITERATION_HELPS': '通过', 'PASS': '通过', 'SUPPORTED': '通过', 'NOT_SUPPORTED': '未通过', 'FAIL': '未通过',
                'INCOMPLETE': '数据不完整'}
    gates = [('T1 迭代效应', 'SA+L24r2', ('base',), summary['T1_iteration'], f'终点成本胜 ≥ {study.WINS}/5，且 AUC 差值中位数 > 0'),
             ('T2 对无 LLM', 'SA+R', ('base',), summary['T2_vs_R'],
              f'终点成本胜 ≥ {study.WINS}/5，且 AUC 降幅中位数 ≥ {100*study.AUC_REDUCTION:.0f}%'),
             ('T3 对 L118', 'SA+L118', ('base', 'minus', 'plus'), summary['T3_vs_L118'],
              f'终点成本胜 ≥ {study.L118_WINS}/11，且 AUC 差值中位数 > 0')]
    by = {(r['variant'], r['seed'], r['arm']): r for r in rows}
    sizes = [sum(1 for (v, _, a) in by if a == study.FINAL_ARM and v in variants) for _, _, variants, _, _ in gates]
    fig, axes = plt.subplots(len(gates), 1, figsize=(7.4, 2.2+.3*sum(sizes)+1.25*len(gates)), sharex=True,
                             gridspec_kw=dict(height_ratios=sizes))
    for ax, (tag, control, variants, result, rule) in zip(axes, gates):
        keys = [(v, s) for (v, s, a) in by if a == study.FINAL_ARM and v in variants]
        for y, (variant, seed) in enumerate(keys):
            a, b = by[(variant, seed, control)], by[(variant, seed, study.FINAL_ARM)]
            ax.plot([a['auc'], b['auc']], [y, y], color=CONNECTOR, linewidth=1.5, zorder=1)
            scatter(ax, control, [a['auc']], [y])
            scatter(ax, study.FINAL_ARM, [b['auc']], [y])
            ax.annotate(outcome(a, b), (1.02, y), xycoords=('axes fraction', 'data'), va='center', fontsize=8, color=INK)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([f'{VARIANT_NAMES[v]} s{s}' for v, s in keys], fontsize=7.5)
        ax.set_ylim(len(keys)-.5, -.5); ax.set_xlim(0, 4)
        ax.annotate('终点成本', (1.02, -.5), xycoords=('axes fraction', 'data'), va='bottom', fontsize=7.5,
                    color=MUTED, annotation_clip=False)
        where = '原工况' if variants == ('base',) else '全部三个工况'
        reduction = result.get('median_auc_reduction')
        ax.set_title(f"{tag}：SA+L24 对 {control}（{where}，{result['pairs']} 对）\n"
                     f"判据：{rule}\n"
                     f"结果：胜 {result['wins']}/{result['pairs']}，AUC 差值中位数 {result['median_auc_difference']:.3f}"
                     f"（降低 {100*reduction:.1f}%）→ {verdicts.get(result['status'], result['status'])}",
                     fontsize=8.5, color=INK, loc='left')
        style(ax); ax.grid(axis='y', visible=False)
    axes[-1].set_xlabel('AUC：运行全程与参考最优解差距的时间平均（百分点，越低越好）', fontsize=8.5)
    order = (study.CONTROL_ARM, study.L118_ARM, study.EARLY_ARM, study.FINAL_ARM)
    handles = [marker_handle(a) for a in order]+[plt.Line2D([], [], color=CONNECTOR, linewidth=1.5)]
    labels = [LABELS[a] for a in order]+['灰色连线：同一配对中两种方法的 AUC 之差']
    fig.tight_layout(rect=(0, .03, 1, .955))
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
    note(fig, '每行是一个配对：同一工况、同一种子，两种方法从同一个初始方案出发。右栏是 SA+L24 的终点成本相对对照方法的胜负，'
              '\n两者终点成本之差不超过成本容差（初始成本的百万分之一，约 0.9–1.0）时记为“平”。', .025)
    save(fig, path)


# 终点差距：每个配对中各方法运行结束时的参考差距。
def final_gap_figure(rows, path):
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    keys = list(dict.fromkeys((r['variant'], r['seed']) for r in rows))
    arms = list(study.BASE_ARMS)
    for k, arm in enumerate(arms):
        points = [(x+(k-(len(arms)-1)/2)*.16, r['reference_gap_percent']) for x, key in enumerate(keys)
                  for r in rows if (r['variant'], r['seed']) == key and r['arm'] == arm]
        scatter(ax, arm, [p[0] for p in points], [p[1] for p in points], size=36)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f'{VARIANT_NAMES[v]}\ns{s}' for v, s in keys], fontsize=7.5)
    ax.set_ylabel('运行结束时与参考最优解的差距（%，越低越好）', fontsize=8.5); ax.set_ylim(bottom=0)
    ax.set_title('各配对的终点差距（原工况 600 秒；负荷下调、上调工况 300 秒）', fontsize=9.5, color=INK)
    style(ax); ax.grid(axis='x', visible=False)
    fig.legend([marker_handle(a) for a in arms], [LABELS[a] for a in arms], loc='upper center', ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .05, 1, .9))
    note(fig, '每一列是一个配对（同一工况、同一种子）；列内从左到右依次是 SA+R、SA+L118、SA+L24r2、SA+L24。'
              '原工况以外的两个工况不运行 SA+L24r2。', .04)
    save(fig, path)


LEGEND_NOTES = '''## 图例说明

**方法（四张图通用，颜色与标记固定）：**
- 橙色方块，点划线：SA+R，在 SA+Benders 上加随机区间算子 R，不调用 LLM。
- 青色三角，虚线：SA+L118，加程序 L118（v1.1.0 发布时称 L3），在 IEEE118 上经两轮生成。
- 蓝色空心圆，点线：SA+L24r2，加程序 L24r2，即 RTS24 训练第 2 轮的最佳程序 P2a，作为早期对照。
- 蓝色实心圆，实线：SA+L24，加程序 L24，即 RTS24 训练 10 轮后的终选程序 P9b。

**F5 训练学习曲线：**
- 横轴为训练轮次，纵轴为验证均分 J，越高越好。
- J 的定义：从固定锚点出发做 15 秒探测，取前 10 秒内成本改善的时间平均，单位为百分点。
- 灰点：每轮生成的单个程序。
- 蓝色阶梯：截至本轮的最佳均分 B_r。实心点表示本轮计为“改进”，即比上一轮至少高 max(0.10, 3%)；空心点表示本轮没有达到改进门槛。
- 橙线：随机算子 R 的验证均分。
- 第 1 轮的两个程序都没有通过静态检查，因此没有点。

**F7 收敛曲线：**
- 三幅分别对应原工况、负荷下调、负荷上调。
- 横轴为计算时间，采用对数刻度，以便看清前 20 秒；纵轴为当前最好方案与参考最优解的差距，越低越好。
- 粗线：同一方法各种子的中位数；细线：单个种子。

**F8 按判据配对的 AUC：**
- 三幅分别对应判据 T1、T2、T3，共用横轴。
- 每行是一个配对：同一工况、同一种子，两种方法从同一个初始方案出发。
- 行内两个点分别是对照方法与 SA+L24 的 AUC，即全程差距的时间平均，越低越好；灰色连线越长，差别越大。
- 右栏“胜 / 平 / 负”：SA+L24 的终点成本相对对照方法的胜负，差距在成本容差以内记为“平”。
- 每幅的标题依次写明判据、结果和结论。

**F6 终点差距：**
- 每一列是一个配对；列内从左到右依次是 SA+R、SA+L118、SA+L24r2、SA+L24。
- 纵轴为运行结束时与参考最优解的差距。
- 负荷下调、负荷上调两个工况不运行 SA+L24r2。

'''


# 与图配套的数据表（表格视图），数值取自冻结的运行与学习曲线。
def tables(curve, rows, path):
    lines = ['# 图表说明与数据', '', LEGEND_NOTES.rstrip(), '', '## 学习曲线数据（验证均分 J）', '',
             '| 轮 | 程序 | 状态 | 验证均分 J | B_r | 改进 |', '|---:|---|---|---:|---:|---|']
    for r in curve['rounds']:
        for i, p in r['programs'].items():
            j = '—' if p['validation'] is None else f"{p['validation']:.3f}"
            b = '—' if r['best_validation'] is None else f"{r['best_validation']:.3f}"
            lines.append(f"| {r['round']} | {i} | {p['status']} | {j} | {b} | {'是' if r['improved'] else '否'} |")
    lines += ['', f"随机算子 R 的验证均分：{curve['R_validation_mean']:.3f}。", '', '## 验证运行数据', '',
              '| 工况 | 种子 | 方法 | 终点参考差距 | AUC | 首次改善 (秒) |', '|---|---:|---|---:|---:|---:|']
    order = [v for v, _ in VARIANTS]
    for r in sorted(rows, key=lambda r: (order.index(r['variant']), r['seed'], study.BASE_ARMS.index(r['arm']))):
        first = '—' if r['first_improvement_seconds'] is None else f"{r['first_improvement_seconds']:.1f}"
        lines.append(f"| {VARIANT_NAMES[r['variant']]} | {r['seed']} | {r['arm']} | {r['reference_gap_percent']:.3f}% | "
                     f"{r['auc']:.3f} | {first} |")
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


# v1.1.0 论文实验（SA 对 SA+L118，ieee118s10 与 rts24 各 5 个种子、600 秒）的同类图；胜负按当时判据：终点差距严格更小为胜。
V110_INSTANCES = (('ieee118s10', 'IEEE118 十场景（ieee118s10）：5 个种子，各 600 秒'),
                  ('rts24', 'RTS24 三场景（rts24）：5 个种子，各 600 秒'))
V110_NAMES = dict(ieee118s10='IEEE118', rts24='RTS24')
V110_ARMS = ('SA', 'SA+L118')


def load_v110(compare_dir):
    rows = [read(p) for p in sorted((compare_dir/'runs').glob('*.json'))]
    return [dict(r, arm=LEGACY.get(r['arm'], r['arm'])) for r in rows]


def v110_outcome(control, treatment):
    a, b = control['gap_percent'], treatment['gap_percent']
    return '胜' if b < a-1e-9 else '平' if abs(b-a) <= 1e-9 else '负'


def v110_convergence(rows, path):
    fig, axes = plt.subplots(1, len(V110_INSTANCES), figsize=(9.4, 3.9), squeeze=False)
    for ax, (name, title) in zip(axes[0], V110_INSTANCES):
        mine = [r for r in rows if r['instance'] == name]
        start = .9*min(r['trajectory'][0][0] for r in mine)
        grid = np.geomspace(start, max(r['seconds'] for r in mine)-5, 800)
        for arm in V110_ARMS:
            curves = np.array([gap_curve(r, grid) for r in mine if r['arm'] == arm])
            s = METHOD_STYLE[arm]
            for c in curves:
                ax.step(grid, c, where='post', color=s['color'], linestyle=s['linestyle'], linewidth=.6, alpha=.3, zorder=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median = np.nanmedian(curves, axis=0)
            ax.step(grid, median, where='post', color=s['color'], linestyle=s['linestyle'], linewidth=2, zorder=3)
        ax.set_xscale('log'); ax.set_xlim(start, grid[-1]*1.05)
        ticks = [t for t in (1, 10, 100) if t >= start]+[grid[-1]]
        ax.set_xticks(ticks); ax.set_xticklabels([f'{t:.0f}' for t in ticks[:-1]]+[f'{grid[-1]+5:.0f}'])
        ax.set_ylim(bottom=0); ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('计算时间（秒，对数刻度）', fontsize=8.5)
        style(ax)
    axes[0][0].set_ylabel('与参考最优解的差距（%，越低越好）', fontsize=8.5)
    handles = [plt.Line2D([], [], color=METHOD_STYLE[a]['color'], linestyle=METHOD_STYLE[a]['linestyle'], linewidth=2)
               for a in V110_ARMS]
    handles += [plt.Line2D([], [], color=MUTED, linewidth=2), plt.Line2D([], [], color=MUTED, linewidth=.6, alpha=.6)]
    labels = [LABELS[a] for a in V110_ARMS]+['粗线：同一方法各种子的中位数', '细线：单个种子']
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .08, 1, .85))
    note(fig, '两种方法在同一算例、同一种子下从同一个初始方案出发；初始化在 IEEE118 十场景上约需 10 秒、RTS24 上约 1 秒，'
              '\n曲线从初始化完成时开始。SA 在全部 10 次运行中初始化之后都没有再改进，因此是水平线。', .07)
    save(fig, path)


def v110_gates(rows, summary, path):
    by = {(r['instance'], r['seed'], r['arm']): r for r in rows}
    fig, axes = plt.subplots(len(V110_INSTANCES), 1, figsize=(7.4, 7.2), sharex=True)
    for ax, (tag, (name, _)) in zip(axes, zip(('M1', 'M2'), V110_INSTANCES)):
        result = summary['criteria'][f'{tag} {name}']
        seeds = sorted({s for (i, s, _) in by if i == name})
        for y, seed in enumerate(seeds):
            a, b = by[(name, seed, 'SA')], by[(name, seed, 'SA+L118')]
            ax.plot([a['auc'], b['auc']], [y, y], color=CONNECTOR, linewidth=1.5, zorder=1)
            scatter(ax, 'SA', [a['auc']], [y])
            scatter(ax, 'SA+L118', [b['auc']], [y])
            ax.annotate(v110_outcome(a, b), (1.02, y), xycoords=('axes fraction', 'data'), va='center', fontsize=8, color=INK)
        ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f'{V110_NAMES[name]} s{s}' for s in seeds], fontsize=7.5)
        ax.set_ylim(len(seeds)-.5, -.5); ax.set_xlim(0, 6)
        ax.annotate('终点差距', (1.02, -.5), xycoords=('axes fraction', 'data'), va='bottom', fontsize=7.5, color=MUTED,
                    annotation_clip=False)
        ax.set_title(f"{tag} {dict(V110_INSTANCES)[name].split('：')[0]}：SA+L118 对 SA（{result['pairs']} 对）\n"
                     f"判据：终点差距更小的种子 ≥ 4/5，且 AUC 降幅中位数 ≥ 10%\n"
                     f"结果：胜 {result['wins']}/{result['pairs']}，AUC 降幅中位数 {100*result['auc_reduction']:.1f}%"
                     f" → {'通过' if result['passed'] else '未通过'}", fontsize=8.5, color=INK, loc='left')
        style(ax); ax.grid(axis='y', visible=False)
    axes[-1].set_xlabel('AUC：600 秒内与参考最优解差距的时间平均（百分点，越低越好）', fontsize=8.5)
    handles = [marker_handle(a) for a in V110_ARMS]+[plt.Line2D([], [], color=CONNECTOR, linewidth=1.5)]
    labels = [LABELS[a] for a in V110_ARMS]+['灰色连线：同一配对中两种方法的 AUC 之差']
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
    note(fig, '每行是一个配对：同一算例、同一种子，两种方法从同一个初始方案出发。右栏是 SA+L118 的终点差距相对 SA 的胜负'
              '\n（严格更小为“胜”）。', .035)
    save(fig, path)


def v110_final_gap(rows, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    keys = [(name, s) for name, _ in V110_INSTANCES for s in sorted({r['seed'] for r in rows if r['instance'] == name})]
    for k, arm in enumerate(V110_ARMS):
        points = [(x+(k-.5)*.22, r['gap_percent']) for x, key in enumerate(keys)
                  for r in rows if (r['instance'], r['seed']) == key and r['arm'] == arm]
        scatter(ax, arm, [p[0] for p in points], [p[1] for p in points], size=40)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f'{V110_NAMES[n]}\ns{s}' for n, s in keys], fontsize=7.5)
    ax.set_ylabel('运行结束时与参考最优解的差距（%，越低越好）', fontsize=8.5); ax.set_ylim(bottom=0)
    ax.set_title('各配对的终点差距（每次运行 600 秒）', fontsize=9.5, color=INK)
    style(ax); ax.grid(axis='x', visible=False)
    fig.legend([marker_handle(a) for a in V110_ARMS], [LABELS[a] for a in V110_ARMS], loc='upper center', ncol=2,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .05, 1, .9))
    note(fig, '每一列是一个配对（同一算例、同一种子）；列内左边是 SA，右边是 SA+L118。', .04)
    save(fig, path)


# L118 的生成过程（2026-10-23，IEEE118 单场景）：数据从当时的 Codex 会话日志重建，见 l118-generation-from-log.json。
GENERATOR_COLORS = dict(R=ORANGE, H='#7a7a7a', L=METHOD_STYLE['SA+L118']['color'])


def l118_generation_figure(data, path):
    groups = [('对照方法', [g for g in data['generators'] if g['kind'] == 'control']),
              ('第 1 轮（只看到对照方法的反馈）', [g for g in data['generators'] if g['round'] == 1]),
              ('第 2 轮（另看到 L0、L1 的代码与训练分数）', [g for g in data['generators'] if g['round'] == 2])]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y, ticks, labels = 0, [], []
    for title, members in groups:
        ax.annotate(title, (.01, y), xycoords=('axes fraction', 'data'), ha='left', va='center', fontsize=8, color=MUTED)
        y += 1
        for g in members:
            color = GENERATOR_COLORS[g['id'][0]]
            if g['validation'] is not None:
                ax.plot([g['train'], g['validation']], [y, y], color=CONNECTOR, linewidth=1.5, zorder=1)
                ax.scatter([g['validation']], [y], s=52, facecolors='white', edgecolors=color, linewidths=1.6, zorder=3)
            ax.scatter([g['train']], [y], s=52, color=color, edgecolors='white', linewidths=.8, zorder=4)
            ticks.append(y); labels.append(g['id'])
            if g['id'] == data['selected']['L']:
                ax.annotate('终选，即现在的 L118', (g['validation'], y), xytext=(10, 0), textcoords='offset points',
                            va='center', fontsize=8, color=INK)
            y += 1
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(y-.4, -.6); ax.set_xlim(0, 2.4)
    ax.set_xlabel('J：从固定锚点出发 15 秒探测中前 10 秒的平均成本改善（百分点，越高越好）', fontsize=8.5)
    ax.set_title('L118（原 L3）的生成过程：IEEE118 单场景，两轮、每轮 2 个程序（2026-10-23）', fontsize=9.5, color=INK)
    style(ax); ax.grid(axis='y', visible=False)
    point = lambda **k: plt.Line2D([], [], linestyle='none', marker='o', markersize=7, **k)
    handles = [point(markerfacecolor=MUTED, markeredgecolor=MUTED), point(markerfacecolor='white', markeredgecolor=MUTED,
               markeredgewidth=1.5), point(markerfacecolor=ORANGE, markeredgecolor=ORANGE),
               point(markerfacecolor=GENERATOR_COLORS['H'], markeredgecolor=GENERATOR_COLORS['H']),
               point(markerfacecolor=GENERATOR_COLORS['L'], markeredgecolor=GENERATOR_COLORS['L'])]
    labels = ['实心：训练均分 J（3 个训练变体 × 2 个锚点）', '空心：验证均分 J（2 个验证变体 × 2 个锚点）',
              '橙色：随机算子 R', '灰色：手写规则 H0–H3', '青色：LLM 程序 L0–L3']
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .1, 1, .88))
    note(fig, '验证只测 R、训练前两名的手写规则和训练前两名的程序，其余没有空心点。L2 的训练分更高，但 L3 的验证分更高（1.892 对 1.843），'
              '\n按规则选 L3。数据从当时的 Codex 会话日志重建，原始结果目录已不存在。', .075)
    save(fig, path)


def l118_test_figure(data, path):
    styles = {'A': ('SA', '#7a7a7a', 'o', True), 'N': ('R', ORANGE, 's', False), 'L': ('L118', METHOD_STYLE['SA+L118']['color'], '^', False),
              'N+SA': ('SA+R', ORANGE, 's', True), 'L+SA': ('SA+L118', METHOD_STYLE['SA+L118']['color'], '^', True)}
    names = {'A': 'SA：只有 SA+Benders', 'N': 'R：只用随机算子，不跑 SA', 'L': 'L118：只用程序 L118，不跑 SA',
             'N+SA': 'SA+R：随机算子与 SA 交替', 'L+SA': 'SA+L118：程序 L118 与 SA 交替'}
    variants = dict(test1='(−0.04, −0.02)', test2='(0.04, 0.02)', test3='(0, −0.04)')
    rows = data['heldout_test']['auc']
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for y, r in enumerate(rows):
        ax.plot([min(r[g] for g in styles), max(r[g] for g in styles)], [y, y], color=CONNECTOR, linewidth=1.2, zorder=1)
        for g, (_, color, marker, filled) in styles.items():
            ax.scatter([r[g]], [y], marker=marker, s=48, zorder=3, linewidths=1.4 if not filled else .8,
                       facecolors=color if filled else 'white', edgecolors='white' if filled else color)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"变体 {r['variant'][-1]} {variants[r['variant']]} s{r['seed']}" for r in rows], fontsize=7.5)
    ax.set_ylim(len(rows)-.5, -.5); ax.set_xlim(left=0)
    ax.set_xlabel('AUC：60 秒内与参考最优解差距的时间平均（百分点，越低越好）', fontsize=8.5)
    ax.set_title('生成时的留出测试：3 个负荷变体 × 2 个种子，每次 60 秒冷启动', fontsize=9.5, color=INK)
    style(ax); ax.grid(axis='y', visible=False)
    handles = [plt.Line2D([], [], linestyle='none', marker=m, markersize=7, markeredgewidth=1.4,
                          markerfacecolor=c if f else 'white', markeredgecolor=c) for _, c, m, f in styles.values()]
    fig.legend(handles, [names[g] for g in styles], loc='upper center', ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, .1, 1, .83))
    note(fig, '实心：与 SA 交替运行；空心：只运行算子、不跑 SA。负荷变体参数 (a, b) 见纵轴。当时的结论：只用 L118 最好'
              '\n（平均终点差距 0.53%），加上 SA 反而更差（SA+L118 对 L118 为 0/6）。数据从当时的 Codex 会话日志重建。', .075)
    save(fig, path)


V110_NOTES = '''## 图例说明

**方法（各图通用，颜色与标记固定，与 v1.2.0 的图一致）：**
- 灰色圆点，实线：SA，只有 SA+Benders（v1.1.0 发布时称 A）。
- 青色三角，虚线：SA+L118，加程序 L118（v1.1.0 发布时称 L+SA，程序称 L3），在 IEEE118 上生成。

**L118-generation 生成过程**（数据从 2026-10-23 的 Codex 会话日志重建，见 l118-generation-from-log.json）：
- 纵轴是 9 个生成器，分三组：对照方法（R、H0–H3）、第 1 轮程序（L0、L1）、第 2 轮程序（L2、L3）。
- 横轴是 J，越高越好。实心点：训练均分；空心点：验证均分。
- 按协议，验证只测 R、训练前两名的手写规则和训练前两名的程序，所以 H1、H2、L0、L1 没有空心点。
- 颜色：橙色是随机算子 R，灰色是手写规则，青色是 LLM 程序。
- L3 被选中，即现在的 L118。

**L118-heldout-test 生成时的留出测试**（同一来源）：
- 每行是一个测试运行，即负荷变体 × 种子；横轴是 60 秒内的 AUC，越低越好。
- 颜色表示算子：灰色是 SA 本身，橙色是随机算子 R，青色是程序 L118。
- 实心表示与 SA 交替运行；空心表示只运行算子、不跑 SA。

**convergence 收敛曲线：**
- 两幅分别对应 IEEE118 十场景与 RTS24 三场景。
- 横轴为计算时间，采用对数刻度；纵轴为当前最好方案与参考最优解的差距，越低越好。
- 粗线：同一方法各种子的中位数；细线：单个种子。
- SA 在初始化之后从未改进，因此是一条水平线。

**paired-auc 按判据配对的 AUC：**
- 两幅分别对应判据 M1（IEEE118 十场景）与 M2（RTS24）。
- 每行是一个配对：同一算例、同一种子。
- 行内两个点分别是 SA 与 SA+L118 的 AUC，即 600 秒内差距的时间平均，越低越好；灰色连线越长，差别越大。
- 右栏“胜 / 平 / 负”：SA+L118 的终点差距相对 SA 的胜负，严格更小为“胜”。

**final-gap 终点差距：**
- 每一列是一个配对，列内左边是 SA，右边是 SA+L118。

'''


def v110_tables(rows, summary, path, generation=None):
    lines = ['# v1.1.0 图表说明与数据', '', V110_NOTES.rstrip(), '']
    if generation:
        fmt = lambda v: '—' if v is None else f'{v:.6f}'
        lines += ['## L118 生成过程（从会话日志重建）', '', f"来源：{generation['source_log']}；{generation['note']}", '',
                  '| 生成器 | 轮次 | 训练均分 J | 验证均分 J |', '|---|---|---:|---:|']
        lines += [f"| {g['id']} | {g['round'] or '对照'} | {fmt(g['train'])} | {fmt(g['validation'])} |" for g in generation['generators']]
        lines += ['', 'LLM 请求：'+'；'.join(f"第 {q['round']} 轮第 {q['attempt']} 次 {q['seconds']:.0f} 秒，{q['status']}"
                                       for q in generation['llm_requests'])+'。', '',
                  '| 留出测试 | 种子 | SA | R | L118 | SA+R | SA+L118 |', '|---|---:|---:|---:|---:|---:|---:|']
        lines += [f"| {r['variant']} | {r['seed']} | {r['A']:.3f} | {r['N']:.3f} | {r['L']:.3f} | {r['N+SA']:.3f} | {r['L+SA']:.3f} |"
                  for r in generation['heldout_test']['auc']]
        lines.append('')
    lines += ['## 判据', '']
    for key, value in summary['criteria'].items():
        lines.append(f"- {key}：胜 {value['wins']}/{value['pairs']}，AUC 降幅中位数 {100*value['auc_reduction']:.1f}%，"
                     f"{'通过' if value['passed'] else '未通过'}。")
    lines += ['', '## 运行数据', '', '| 算例 | 种子 | 方法 | 终点参考差距 | AUC |', '|---|---:|---|---:|---:|']
    for r in sorted(rows, key=lambda r: ([n for n, _ in V110_INSTANCES].index(r['instance']), r['seed'], V110_ARMS.index(r['arm']))):
        lines.append(f"| {V110_NAMES[r['instance']]} | {r['seed']} | {r['arm']} | {r['gap_percent']:.3f}% | {r['auc']:.3f} |")
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def v110_main(compare_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = load_v110(compare_dir)
    summary = read(compare_dir/'summary.json')
    v110_convergence(rows, output/'convergence')
    v110_gates(rows, summary, output/'paired-auc')
    v110_final_gap(rows, output/'final-gap')
    source = output/'l118-generation-from-log.json'
    generation = read(source) if source.exists() else None
    if generation:
        l118_generation_figure(generation, output/'L118-generation')
        l118_test_figure(generation, output/'L118-heldout-test')
    v110_tables(rows, summary, output/'data.md', generation)
    print('\n'.join(sorted(p.name for p in output.iterdir())))


def main(training, evaluation_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    curve = read(training/'learning-curve.json')
    rows = load_runs(evaluation_dir)
    summary = study.summarize(rows)
    learning_figure(curve, output/'F5-learning-curve')
    convergence_figure(rows, output/'F7-convergence')
    gates_figure(rows, summary, output/'F8-paired-auc')
    final_gap_figure(rows, output/'F6-final-gap')
    tables(curve, rows, output/'data.md')
    print('\n'.join(sorted(p.name for p in output.iterdir())))


if __name__ == '__main__':
    if sys.argv[1] == '--v110':
        v110_main(*(Path(a).resolve() for a in sys.argv[2:4]))
    else:
        main(*(Path(a).resolve() for a in sys.argv[1:4]))
