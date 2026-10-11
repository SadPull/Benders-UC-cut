"""Score only immutable in-budget checkpoints with independent audit results."""
import csv
import json
from pathlib import Path

from .artifacts import read_jsonl, save, scored_trace
from .audit import verify_trace


METHODS = dict(A='作者传统 3-bin Benders', B='全天整数认证和全天及小时割',
               C='B 加 SA 分块搜索和协调', D='C 加上下界双通道 LLM')


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def reaches(row, tolerance):
    return row['objective']-row['lower_bound'] <= 1e-5+tolerance*max(1., abs(row['objective']))


def summarize_arm(directory, arm, budget):
    folder = directory/arm
    state = read(folder/'state.json', {})
    rows = scored_trace(folder, budget)
    verify_trace(rows)
    audit = read(directory/'audits'/arm/'audit.json', {})
    audited = bool(rows and audit.get('passed') and audit.get('final') == rows[-1])
    result = read(folder/'result.json', {})
    stopped = result.get('status', state.get('status', 'NOT_STARTED'))
    status = ('NO_SCORED_CHECKPOINT' if state.get('numerical_runs_started') else 'NOT_STARTED') if not rows else (
        ('CONVERGED' if reaches(rows[-1], .001) else 'NOT_CONVERGED') if audited else 'PENDING_AUDIT')
    summary = dict(arm=arm, method=METHODS[arm], stop_reason=stopped, audit_passed=audited, status=status,
        scored_checkpoints=len(rows), peak_rss_bytes=state.get('peak_rss_bytes'),
        process_wall_seconds=state.get('wall_seconds'), reached_1_percent_seconds=None,
        reached_0_1_percent_seconds=None, snapshots={}, objective=None, lower_bound=None, relative_gap=None)
    if not rows:
        return summary
    final = rows[-1]
    summary.update({k: final.get(k) for k in ('objective', 'lower_bound', 'relative_gap', 'sa_calls', 'llm_calls')})
    summary['last_scored_seconds'] = final['elapsed_seconds']
    for name, tolerance in (('reached_1_percent_seconds', .01), ('reached_0_1_percent_seconds', .001)):
        summary[name] = next((r['elapsed_seconds'] for r in rows if reaches(r, tolerance)), None) if audited else None
    for seconds in (300, 600, 1800, 3600):
        snapshot = next((r for r in reversed(rows) if r['elapsed_seconds'] <= seconds), None)
        summary['snapshots'][str(seconds)] = ({k: snapshot[k] for k in
            ('objective', 'lower_bound', 'relative_gap', 'elapsed_seconds')} if snapshot else None)
    incumbent = read(folder/final['checkpoint'], {})
    summary['validation'] = incumbent.get('validation')
    summary['global_cuts'] = sum(r.get('elapsed_seconds', budget+1) <= budget for r in read_jsonl(folder/'certificate_cuts.jsonl'))
    summary['hourly_cuts'] = sum(r.get('elapsed_seconds', budget+1) <= budget for r in read_jsonl(folder/'hourly_certificate_cuts.jsonl'))
    summary['timings'] = incumbent.get('timings', {})
    return summary


def conclusion(arms):
    if not all(a['audit_passed'] for a in arms.values()):
        return '四组正式成绩及独立审计尚未全部完成，当前不能判断项目或 LLM 是否改善收敛。'
    b, c, d = (arms[a] for a in 'BCD')
    if d['status'] != 'CONVERGED':
        return 'D 未达到 0.1% 有效上下界间隙，本轮最终收敛目标未完成。'
    if b['status'] == 'CONVERGED' and b['reached_0_1_percent_seconds'] <= d['reached_0_1_percent_seconds']:
        return 'B 已更早达到门槛，完整方法的收敛收益主要已由整数认证及有效割实现，不能归功于 LLM。'
    if c['status'] != 'CONVERGED':
        return 'D 达标而 C 未达标，本实例支持双通道 LLM 帮助实现收敛。'
    if d['reached_0_1_percent_seconds'] >= c['reached_0_1_percent_seconds']:
        return 'C 和 D 均达标，但 D 没有更快，不能宣称 LLM 加速。'
    return 'C 和 D 均达标，D 更早达到 0.1% 门槛；结论限定于本实例和种子 7。'


def write_report(directory, protocol):
    directory = Path(directory)
    arms = {a: summarize_arm(directory, a, protocol.time_limit) for a in 'ABCD'}
    state = read(directory/'state.json', {})
    preflight = read(directory/'preflight.json', {})
    summary = dict(status=state.get('status', 'NOT_STARTED'), protocol=vars(protocol), arms=arms,
        conclusion=conclusion(arms), preflight_status=preflight.get('status', 'NOT_STARTED'),
        passed_checks=preflight.get('passed_checks', []), resource_gate=read(directory/'resource_gate.json', {}))
    save(directory/'summary.json', summary)
    lines = ['# SMS100 首轮实验进度与结果', '', summary['conclusion'], '',
        f"固定输入为 `{protocol.base_file}`，100 台机组、24 小时、原始训练集第 1 至 100 个日场景，"
        f"batch {protocol.batch}，force 1.0，种子 {protocol.seed}。四组均为单线程，每组预算 {protocol.time_limit:g} 秒。", '',
        f"当前流程状态：**{summary['status']}**。数值预检状态：**{summary['preflight_status']}**。", '']
    if summary['status'] == 'BLOCKED_MEMORY':
        gate = state if 'available_bytes' in state else summary['resource_gate']
        lines += [f"可用内存 {gate.get('available_bytes', 0)/1024**3:.3f} GiB，启动要求至少 "
                  f"{gate.get('required_bytes', 4*1024**3)/1024**3:.3f} GiB。资源门禁已阻止数值启动。", '']
    if state.get('error'):
        lines += [f"停止阶段 `{state.get('stage')}`：{state['error']}", '']
    lines += ['| 组别 | 方法 | 状态 | UB | LB | gap | 到 1% 秒 | 到 0.1% 秒 | 审计 |',
              '|---|---|---|---:|---:|---:|---:|---:|---|']
    number = lambda v: '—' if v is None else f'{v:.6f}'
    for a, row in arms.items():
        gap = '—' if row['relative_gap'] is None else f"{100*row['relative_gap']:.6f}%"
        lines.append(f"| {a} | {row['method']} | {row['status']} | {number(row['objective'])} | "
            f"{number(row['lower_bound'])} | {gap} | {number(row['reached_1_percent_seconds'])} | "
            f"{number(row['reached_0_1_percent_seconds'])} | {'通过' if row['audit_passed'] else '待完成'} |")
    lines += ['', '共同连续松弛及初始方案分别计入每组预算。正式运行期间的方案、割、LLM 回复和检查点各组独立。'
        'gap 使用 `(UB-LB)/max(1,abs(UB))`；UB 须经作者模型全部 100 场景重算，LB 来自有效数学认证。', '',
        '共同连续松弛采用 Gurobi 单线程屏障法，另外重算其对偶证书。首次自动算法预检未完成的记录及源码'
        '保存在 `attempts/preflight-02`，其耗时属于预检，不作为任何正式组的成绩。', '',
        '## 预算内进展', '', '| 组别 | 秒数 | UB | LB | gap | 检查点秒数 |', '|---|---:|---:|---:|---:|---:|']
    for a, arm in arms.items():
        for seconds in (300, 600, 1800, 3600):
            row = arm['snapshots'].get(str(seconds))
            lines.append(f'| {a} | {seconds} | — | — | — | — |' if row is None else
                f"| {a} | {seconds} | {number(row['objective'])} | {number(row['lower_bound'])} | "
                f"{100*row['relative_gap']:.6f}% | {number(row['elapsed_seconds'])} |")
    lines += ['', '## 验证证据', '',
        '检查点只有在预算内完成原子保存并核对哈希后才进入成绩。超时后的求解、LLM 回复及最终保存不会回填一小时成绩。', '',
        '数值预检须覆盖作者和 Python 模型费用对照、小规模割穷举、核心回归、八个原规模 SA 时间块及一次协调、'
        '真实上下界 LLM 请求和关闭通道检查。共同初始下界另外按弱对偶重算，初始方案由作者模型独立重算。', '',
        '来源冻结见 [upstream.json](upstream.json)、[environment.json](environment.json) 和 '
        '[source_sha256.json](source_sha256.json)。详细成绩见 [summary.json](summary.json)。', '',
        '## 继续运行', '', '在项目根目录执行：', '', '```powershell',
        r'.\.benchmark\azema\python\Scripts\python.exe -B run_benchmark.py --phase all', '```', '',
        '该入口先执行全部预检，然后按 A、B、C、D 顺序运行并独立审计。已启动的正式组不会被覆盖或延长。'
        '首轮结束交付报告，后续批次及多种子复核另行安排。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (directory/'bounds.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('arm', 'elapsed_seconds', 'objective', 'lower_bound', 'relative_gap'))
        writer.writeheader()
        for a in 'ABCD':
            for row in scored_trace(directory/a, protocol.time_limit):
                writer.writerow(dict(arm=a, **{k: row[k] for k in writer.fieldnames if k != 'arm'}))
