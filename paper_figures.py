"""论文补充实验的静态图：收敛曲线、QUBO 位数与 L3 案例。"""
# 配色取参考调色板前三个类别槽位（三色在全部两两配对下通过色觉校验），并用线型做二次编码以便黑白打印。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

import paper_pilot as pilot

INK, MUTED, GRID, OFF = '#0b0b0b', '#52514e', '#e4e3df', '#f1f0ed'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
ARM_STYLE = {'A': dict(color=BLUE, linestyle='-', label='A: SA+Benders (S1)'),
             'L+SA': dict(color=ORANGE, linestyle='--', label='L+SA: + offline program L3 (S3)')}
TITLES = dict(ieee118s10='IEEE118, 54 units x 24 h x 10 scenarios', rts24='RTS24, 32 units x 24 h x 3 scenarios',
              ieee118='IEEE118, 54 units x 24 h x 1 scenario', rcuc20='RCUC20, 20 units x 24 h x 1 scenario')


def style(ax):
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# 一次运行的参考差距阶梯函数在网格上的取值；首个可行方案之前为 NaN。
def gap_curve(run, grid):
    times = np.array([t for t, _ in run['trajectory']])
    gaps = np.array([100*(c-run['reference_cost'])/abs(run['reference_cost']) for _, c in run['trajectory']])
    index = np.searchsorted(times, grid, side='right') - 1
    return np.where(index >= 0, gaps[np.clip(index, 0, None)], np.nan)


def convergence_figure(runs, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(3.6*len(names), 2.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        mine = [r for r in runs if r['instance'] == name and r['trajectory']]
        horizon = max(r['seconds'] for r in mine) - 5
        grid = np.linspace(0, horizon, 600)
        for arm, s in ARM_STYLE.items():
            curves = np.array([gap_curve(r, grid) for r in mine if r['arm'] == arm])
            if not len(curves):
                continue
            for c in curves:
                ax.step(grid, c, where='post', color=s['color'], linestyle=s['linestyle'], linewidth=.7, alpha=.35)
            ax.plot(grid, np.median(curves, axis=0), color=s['color'], linestyle=s['linestyle'], linewidth=2,
                    label=f"{s['label']}, median of {len(curves)} seeds")
        ax.set_title(TITLES.get(name, name), fontsize=9, color=INK)
        ax.set_xlabel('Wall time (s)', fontsize=8)
        ax.set_ylabel('Gap to reference (%)', fontsize=8)
        ax.set_ylim(bottom=0)
        style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.08))
    fig.tight_layout()
    save(fig, path)


def bits_figure(runs, path):
    blocks = sorted({s['block'] for r in runs for s in r['qubo_sizes']})
    if not runs or not blocks:
        return
    fig, axes = plt.subplots(len(runs), len(blocks), figsize=(3.4*len(blocks), 2.6*len(runs)), squeeze=False)
    for row, run in zip(axes, runs):
        for ax, block in zip(row, blocks):
            sizes = [s for s in run['qubo_sizes'] if s['block'] == block]
            k = [s['master'] for s in sizes]
            fine = [pilot.standard_bits(s, pilot.THETA_BITS) for s in sizes]
            ax.plot(k, [s['ours'] for s in sizes], color=BLUE, linewidth=2, label='This work (tightest-cut cost)')
            ax.plot(k, [f['same_cuts'] for f in fine], color=ORANGE, linestyle='--', linewidth=2,
                    label='Standard encoding of the same cuts (lower bound)')
            ax.plot(k, [f['single_cut'] for f in fine], color=AQUA, linestyle=':', linewidth=2,
                    label='Standard single-cut Benders (lower bound)')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{TITLES.get(run['instance'], run['instance'])}\nblock {block} (seed {run['seed']})",
                         fontsize=8, color=INK)
            ax.set_xlabel('Master solve in this block', fontsize=8)
            ax.set_ylabel('QUBO variables', fontsize=8)
            style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc='upper center', ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(.5, 0))
    save(fig, path)


def case_figure(row, d, path):
    before, after = np.asarray(row['base_u']), np.asarray(row['u'])
    units = [i-1 for i in row['changed_units']]
    hours = np.arange(1, before.shape[1]+1)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 1.2+.42*2*len(units)+3.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[max(1., .5*len(units)), 1, 1]))
    ax = axes[0]
    labels = []
    for k, i in enumerate(units):
        for j, (u, color, tag) in enumerate(((before[i], BLUE, 'before'), (after[i], ORANGE, 'after'))):
            y = -(2*k+j)
            labels.append((y, f'G{i+1} {tag}'))
            for t in range(len(hours)):
                changed = before[i, t] != after[i, t]
                ax.add_patch(plt.Rectangle((t+.55, y-.4), .9, .8, facecolor=color if u[t] else OFF,
                                           edgecolor=INK if changed else 'white', linewidth=1.2 if changed else 2))
    ax.set_yticks([y for y, _ in labels]); ax.set_yticklabels([s for _, s in labels], fontsize=8)
    ax.set_ylim(min(y for y, _ in labels)-.6, .6)
    ax.set_title('Commitment of the changed units (filled = on; outlined = changed hour)', fontsize=8, color=INK)
    style(ax); ax.grid(False)
    axes[1].plot(hours, row['expected_load'], color=INK, linewidth=2)
    axes[1].set_ylabel('MW', fontsize=8)
    axes[1].set_title('Probability-weighted load', fontsize=8, color=INK)
    axes[2].step(hours, row['expected_price'], where='mid', color=INK, linewidth=2)
    axes[2].set_ylabel('$/MWh', fontsize=8)
    axes[2].set_title('Probability-weighted marginal price at the stalled schedule', fontsize=8, color=INK)
    axes[2].set_xlabel('Hour', fontsize=8)
    for a in axes[1:]:
        style(a)
    axes[2].set_xticks(hours[::2])
    share = 100*row['delta']/abs(row['base_objective'])
    fig.suptitle(f"{TITLES.get(row['instance'], row['instance'])}, seed {row['seed']}: first improving L3 candidate "
                 f"(call {row['calls']}), cost change {row['delta']:,.0f} ({share:.3f}%)", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, path)
