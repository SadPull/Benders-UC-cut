"""Fixed RTS-24 pilot: independent preflight, one 600-second run, then audit."""
from time import perf_counter, time
PROCESS_STARTED, PROCESS_UNIX = perf_counter(), time()
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import gurobipy as gp
import numpy as np

from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.hourly_bounds import HourlyBounds
from suc_benders.model import fingerprint, model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import pack, unpack, validate
from .azema import GUROBI_VERSION, ROOT
from .azema.artifacts import Recorder, memory_gate, read_jsonl, save, scored_trace, sha256, snapshot, source_hashes
from .azema.audit import verify_cut, verify_trace
from .azema.controller import Stopped, require_finished, worker
from .azema.process import run_process
from .azema.protocol import environment


def options(directory):
    private = {'kaiwu_user_id', 'kaiwu_sdk_code', 'llm_url', 'llm_api_key'}
    o = load_options(dict(data_file=str(directory/'inputs/units.json'),
        scenario_file=str(directory/'inputs/stochastic.json'), bound_mode='integer_benders',
        sa_enabled=True, llm_enabled=True, llm_model='gpt-6-astra',
        llm_agent_directory=str(directory/'agent_bridge'), llm_patience=3, llm_max_calls=12,
        llm_timeout=600, seed=7, time_limit=600, tolerance=.001, block_hours=3,
        max_outer=20, max_inner=20, bound_slice_seconds=30, verbose=True))
    return {key: value for key, value in asdict(o).items() if key not in private}


def binding(directory):
    return dict(source=source_hashes(), environment=environment(), options=options(directory),
        inputs={name: sha256(directory/'inputs'/name) for name in ('units.json', 'stochastic.json', 'provenance.json')})


def quadratic_dual(m, prices):
    prices = np.asarray(prices)
    if prices.shape != m.rhs.shape or not np.isfinite(prices).all() or (prices[~m.eq] < 0).any():
        raise ValueError('Invalid initial dual certificate')
    c = m.obj+np.asarray(m.A.T @ prices)
    z = np.where(c < 0, m.ub, m.lb).copy()
    curved = m.q > 0
    z[curved] = np.clip(-c[curved]/(2*m.q[curved]), m.lb[curved], m.ub[curved])
    return float(np.sum(c*z+m.q*z*z)-m.rhs @ prices)


class PilotRecorder(Recorder):
    audit_cuts = True

    def relaxation(self, m, value, elapsed):
        target = self.directory/'initial_prices.npz'
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, prices=value.prices)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        save(self.directory/'initial_bound.json', dict(lower_bound=value.lower,
            model_fingerprint=fingerprint(m), arrays_sha256=sha256(target), elapsed_seconds=elapsed))


def case(inputs):
    d, scenarios = load_data(inputs/'units.json', inputs/'stochastic.json')
    if (d.n_units, d.horizon, len(scenarios), d.reserve_fraction) != (32, 24, 3, .05):
        raise ValueError('RTS-24 pilot input shape or reserve changed')
    return model(d, scenarios)


def fixed_check(m, u, expected, seconds):
    x = pack(m.d, np.asarray(u, float))
    m.lb[m.binary] = x; m.ub[m.binary] = x
    result = optimize(m, seconds)
    if result.status != 'OPTIMAL':
        raise ValueError('Fresh fixed-commitment QP did not solve optimally')
    solved = unpack(m, result.x)
    validation = validate(m.d, m.scenarios, solved)
    if abs(result.objval-expected) > 1e-4:
        raise ValueError('Fresh QP cost differs from recorded cost')
    return dict(objective=result.objval, validation=validation, scenarios=solved.scenarios,
        fixed_cost=result.objval-sum(s.probability*z.objective for s, z in zip(m.scenarios, solved.scenarios)))


def initial_audit(job):
    from .azema.worker import read_common
    folder = Path(job['directory']); m = case(Path(job['inputs']))
    common = read_common(job['common_directory'])
    if common['model_fingerprint'] != fingerprint(m):
        raise ValueError('Initial model fingerprint mismatch')
    lower = quadratic_dual(m, common['relaxation_prices'])
    if common['lower_bound'] > lower+1e-5:
        raise ValueError('Initial lower bound exceeds quadratic dual value')
    checked = fixed_check(m, common['u'], common['objective'], job['time_limit'])
    save(folder/'check.json', dict(passed=True, independent_lower_bound=lower, **checked))


def audit_records(folder, m, budget, initial_lower):
    rows = scored_trace(folder, budget)
    if not rows:
        raise ValueError('No scored checkpoint')
    verify_trace(rows)
    signature = fingerprint(m); full = partition(m, m.d.horizon)[0][0]; hourly = HourlyBounds(full)
    cuts, hour_cuts = read_jsonl(folder/'certificate_cuts.jsonl'), read_jsonl(folder/'hourly_certificate_cuts.jsonl')
    for cut in cuts:
        if cut['elapsed_seconds'] <= budget:
            verify_cut(full, cut)
    for cut in hour_cuts:
        if cut['elapsed_seconds'] <= budget:
            verify_cut(hourly.parts[cut['hour']-1].b, dict(cut, optimality=True))
    native = read_jsonl(folder/'certification_history.jsonl')
    schedules = {}
    for row in rows:
        state = json.loads((folder/row['checkpoint']).read_text(encoding='utf-8'))
        if state['model_fingerprint'] != signature:
            raise ValueError('Scored model fingerprint mismatch')
        offsets = state['offsets']
        bounds = [initial_lower]
        for evidence in native[:offsets['certification_history']]:
            if evidence.get('bound_used'):
                if (evidence['status'] not in ('OPTIMAL', 'TIME_LIMIT', 'INTERRUPTED', 'USER_OBJ_LIMIT')
                        or evidence['cuts'] > offsets['certificate_cuts']
                        or evidence['hourly_cuts'] > offsets['hourly_certificate_cuts']
                        or not np.isfinite(evidence['master_bound'])):
                    raise ValueError('Invalid saved native lower-bound evidence')
                bounds.append(evidence['master_bound'])
        if row['lower_bound'] > max(bounds)+1e-5:
            raise ValueError('Scored lower bound exceeds its evidence')
        key = np.asarray(state['u'], np.uint8).tobytes()
        schedules[key] = (state['u'], row['objective'])
    return rows, list(schedules.values()), dict(global_cuts=len(cuts), hourly_cuts=len(hour_cuts))


def audit(job):
    folder, run = Path(job['directory']), Path(job['run_directory'])
    m = case(Path(job['inputs'])); signature = fingerprint(m)
    initial = json.loads((run/'initial_bound.json').read_text())
    if initial['model_fingerprint'] != signature or sha256(run/'initial_prices.npz') != initial['arrays_sha256']:
        raise ValueError('Initial QP certificate fingerprint or hash mismatch')
    lower = quadratic_dual(m, np.load(run/'initial_prices.npz', allow_pickle=False)['prices'])
    if initial['lower_bound'] > lower+1e-5:
        raise ValueError('Initial QP lower bound is invalid')
    rows, schedules, counts = audit_records(run, m, 600., initial['lower_bound'])
    checked = []
    started = perf_counter()
    for index, (u, cost) in enumerate(schedules):
        value = fixed_check(m, u, cost, job['time_limit']-(perf_counter()-started))
        save(folder/f'incumbent-{index+1:03d}.json', dict(u=u, **value))
        checked.append(dict(objective=value['objective'], validation=value['validation']))
    save(folder/'audit.json', dict(passed=True, final=rows[-1], model_fingerprint=signature,
        initial_dual_value=lower, incumbents=checked, scenarios=3, **counts,
        lower_bound_audit='native_Gurobi_bound_with_algebraically_verified_quadratic_cuts'))


def worker_main(job):
    directory = Path(job['directory'])
    gate = memory_gate(job.get('estimated_peak_bytes', 0))
    if not gate['passed']:
        save(directory/'state.json', dict(status='BLOCKED_MEMORY', numerical_runs_started=0))
        return 3
    if '.'.join(map(str, gp.gurobi.version())) != GUROBI_VERSION:
        raise ValueError('Use the isolated Gurobi 12.0.2 environment')
    if job['action'] == 'run':
        from run_suc import run_suc
        started = PROCESS_STARTED-max(0., PROCESS_UNIX-job['launch_epoch_seconds'])
        result = run_suc(job['options'], progress=PilotRecorder(directory, 600), started_at=started)
        save(directory/'result.json', result)
    elif job['action'] == 'initial_audit':
        initial_audit(job)
    elif job['action'] == 'audit':
        audit(job)
    else:
        raise ValueError('Unknown RTS-24 pilot action')
    return 0


def pilot_worker(action, folder, seconds, **values):
    job = dict(action=action, directory=str(folder), time_limit=seconds, **values)
    save(folder/'job.json', job)
    state = run_process([sys.executable, '-B', '-m', 'benchmarks.rts24', str(folder/'job.json')],
                        folder, seconds, estimated_peak_bytes=values.get('estimated_peak_bytes', 0))
    if action != 'run' or state['status'] == 'BLOCKED_MEMORY':
        require_finished(state, action)
    return state


def run_preflight(directory):
    evidence = dict(status='RUNNING', binding=binding(directory), passed_checks=[], estimated_peak_bytes=0)
    save(directory/'preflight.json', evidence)
    stages = directory/'preflight'; common = stages/'common'

    def passed(name, state):
        evidence['passed_checks'].append(name)
        evidence['estimated_peak_bytes'] = max(evidence['estimated_peak_bytes'], state['peak_rss_bytes'])
        save(directory/'preflight.json', evidence)

    xml = stages/'regression/pytest.xml'
    state = run_process([sys.executable, '-B', '-m', 'pytest', '-q', '-m', 'not integration',
        '-p', 'no:cacheprovider', f'--junitxml={xml}'], stages/'regression', 1800)
    require_finished(state, 'core_regression')
    if any(int(s.get(k, 0)) for s in ET.parse(xml).getroot().iter('testsuite') for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Core regression failed or skipped checks')
    passed('core_regression', state)
    state = worker('common', common, 600, data_file=str(directory/'inputs/units.json'),
                   scenario_file=str(directory/'inputs/stochastic.json'))
    passed('independent_preflight_initialization', state)
    state = pilot_worker('initial_audit', stages/'initial_audit', 600,
                         inputs=str(directory/'inputs'), common_directory=str(common))
    passed('quadratic_initial_bounds', state)
    state = worker('sa', stages/'sa', 600, common_directory=str(common),
                   options=dict(options(directory), llm_enabled=False))
    sampling = json.loads((stages/'sa/check.json').read_text())
    if not sampling['passed'] or sampling['real_sa_calls'] != 8 or sampling['coordination_count'] != 1 or sampling['llm_calls']:
        raise ValueError('Eight-block SA/coordination gate failed')
    passed('eight_sa_blocks_and_coordination', state)
    state = worker('llm', stages/'llm', 1800, common_directory=str(common), options=options(directory))
    receipts = json.loads((stages/'llm/llm_receipts.json').read_text())
    for channel in ('upper', 'lower'):
        if not any(r['channel'] == channel and r['status'] == 'OK' and r.get('numerical_evaluation_completed')
                   and r.get('transport') == 'codex_subagent' and r.get('agent_id')
                   and r.get('response_model') == 'gpt-6-astra' and r.get('reasoning_effort') == 'max' for r in receipts):
            raise ValueError('Missing real numerically evaluated subagent channel: '+channel)
    m = case(directory/'inputs')
    initial = json.loads((common/'common.json').read_text())
    audit_records(stages/'llm', m, 1800, initial['lower_bound'])
    passed('real_upper_and_lower_subagents', state)
    evidence['status'] = 'PASSED'
    save(directory/'preflight.json', evidence)


def write_report(directory):
    from collections import Counter
    read = lambda p, default: json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
    state = read(directory/'state.json', {})
    pre = read(directory/'preflight.json', {})
    run = directory/'run'; rows = scored_trace(run, 600); verify_trace(rows)
    audited = read(directory/'audit/audit.json', {})
    final = rows[-1] if rows else None
    valid = bool(final and audited.get('passed') and audited['final'] == final)
    result = read(run/'result.json', {})
    summary = dict(status=state.get('status'), preflight_status=pre.get('status'), final=final,
        audit_passed=valid, target_met=bool(valid and final['objective']-final['lower_bound'] <=
            1e-5+.001*max(1, abs(final['objective']))), budget_seconds=600,
        stop_reason=result.get('status', read(run/'state.json', {}).get('status', 'NOT_STARTED')),
        formal_llm_calls=final.get('llm_calls', 0) if final else 0,
        preflight_llm_calls=len(read(directory/'preflight/llm/llm_receipts.json', [])),
        process=read(run/'state.json', {}), audit=audited)
    summary['result_status'] = ('CONVERGED' if summary['target_met'] else 'NOT_CONVERGED') if valid else 'PENDING_AUDIT'
    summary['preflight_wall_seconds'] = {name: read(directory/'preflight'/name/'state.json', {}).get('wall_seconds')
        for name in ('regression', 'common', 'initial_audit', 'sa', 'llm')}
    summary['audit_wall_seconds'] = read(directory/'audit/state.json', {}).get('wall_seconds')
    summary['resource_gate'] = read(run/'resource_gate.json', {})
    regression = directory/'preflight/regression/pytest.xml'
    summary['core_regression_tests'] = sum(int(s.get('tests', 0)) for s in ET.parse(regression).getroot().iter('testsuite')) if regression.exists() else None
    if final:
        checkpoint = read(run/final['checkpoint'], {})
        stream = lambda name: read_jsonl(run/(name+'.jsonl'))[:checkpoint['offsets'].get(name, 0)]
        sampling = stream('sampling')
        proposals = [p for p in stream('proposals') if p.get('origin') == 'sa']
        hamming = [p['hamming_u'] for p in proposals if 'hamming_u' in p]
        summary.update(timings=checkpoint['timings'], complete_rounds=checkpoint['complete_rounds'],
            coordination_count=checkpoint['coordination_count'], scored_checkpoints=len(rows),
            sa=dict(raw_candidates=sum(p.get('raw_candidates', p['candidates']) for p in sampling),
                raw_discrete_feasible=sum(p['raw_discrete_feasible'] for p in sampling),
                processed_candidates=len(proposals), outcomes=dict(Counter(p.get('outcome') for p in proposals)),
                repair_hamming_range=[min(hamming), max(hamming)] if hamming else None))
        summary['upper_decrease_by_source'] = {}
        for evaluation in stream('evaluations'):
            before = evaluation.get('incumbent_before')
            if before is not None and np.isfinite(before):
                decrease = before-evaluation['incumbent']
                if decrease > 1e-5:
                    source = evaluation['source']
                    summary['upper_decrease_by_source'][source] = summary['upper_decrease_by_source'].get(source, 0)+decrease
        requests = [r for r in read(run/'llm_receipts.json', []) if r.get('phase') == 'formal'
            and r['started_seconds'] <= final['elapsed_seconds']]
        summary['formal_llm_calls_by_channel'] = dict(Counter(r['channel'] for r in requests))
        summary['formal_receipts'] = [{k: r.get(k) for k in ('call_id', 'channel', 'status', 'agent_id',
            'response_model', 'reasoning_effort', 'request_sha256', 'reply_sha256', 'seconds',
            'consumed_seconds', 'format_valid', 'accepted', 'new_global_cuts', 'direct_incumbent_decrease',
            'numerical_evaluation_completed')} for r in requests
            if r.get('consumed_seconds', final['elapsed_seconds']) <= final['elapsed_seconds']]
        summary['snapshots'] = {str(seconds): next((r for r in reversed(rows) if r['elapsed_seconds'] <= seconds), None)
            for seconds in (60, 300, 600)}
        summary['reached_1_percent_seconds'] = next((r['elapsed_seconds'] for r in rows if r['relative_gap'] <= .01), None) if valid else None
        summary['reached_0_1_percent_seconds'] = next((r['elapsed_seconds'] for r in rows if
            r['objective']-r['lower_bound'] <= 1e-5+.001*max(1., abs(r['objective']))), None) if valid else None
        if valid:
            solutions = sorted((directory/'audit').glob('incumbent-*.json'))
            if solutions:
                path = solutions[-1]
                solution = read(path, {})
                summary['final_solution'] = dict(artifact=path.relative_to(directory).as_posix(),
                    fixed_cost=solution['fixed_cost'], validation=solution['validation'],
                    scenario_costs=[{k: s[k] for k in ('name', 'probability', 'objective')} for s in solution['scenarios']])
    reference = read(directory/'reference/result.json', {})
    if valid and reference:
        manifest = read(directory/'reference/manifest.json', {})
        if manifest['model_fingerprint'] != audited['model_fingerprint'] or not reference['checked']['validation']['feasible']:
            raise ValueError('Reference model or physical validation mismatch')
        cost = reference['checked']['objective']
        summary['reference_comparison'] = dict(status=reference['status'], objective=cost,
            lower_bound=reference['lower_bound'], relative_gap=reference['relative_gap'],
            solve_wall_seconds=reference['solve_wall_seconds'], fixed_cost=reference['checked']['fixed_cost'],
            expected_fuel_cost=cost-reference['checked']['fixed_cost'],
            pilot_minus_reference=final['objective']-cost,
            pilot_reference_cost_gap=(final['objective']-cost)/abs(cost), charged_to_formal_pilot=False)
    save(directory/'summary.json', summary)
    lines = ['# RTS24 完整方法短测', '',
        f"流程状态：**{summary['status']}**。预检：**{summary['preflight_status']}**。独立审计：**{'通过' if valid else '待完成'}**。", '',
        '32 台机组、24 小时、三个场景（倍率 0.90/1.00/1.05，概率 0.25/0.50/0.25），备用 5%。'
        '保留 RTS24 派生聚合模型的二次燃料费用和固定启动费用；种子 7，数值单线程。', '',
        '正式运行预算 600 秒，包含进程启动、输入、SDK 授权、初始化、求解、子代理等待和在线保存。'
        '预检和独立审计单独计时。', '']
    if final:
        lines += [f"最终预算内 UB **{final['objective']:.6f}**，LB **{final['lower_bound']:.6f}**，"
            f"gap **{100*final['relative_gap']:.6f}%**，检查点时间 **{final['elapsed_seconds']:.3f} 秒**。", '',
            f"0.1% 目标：**{'达到' if summary['target_met'] else '尚未达到'}**。停止原因：`{summary['stop_reason']}`。", '',
            f"正式 SA 调用 {final['sa_calls']} 次，正式 LLM 请求 {summary['formal_llm_calls']} 次。", '']
        lines += ['| 预算位置（秒） | UB | LB | gap | 最后检查点（秒） |',
            '|---:|---:|---:|---:|---:|']
        for seconds, row in summary['snapshots'].items():
            if row:
                lines += [f"| {seconds} | {row['objective']:.6f} | {row['lower_bound']:.6f} | "
                    f"{100*row['relative_gap']:.6f}% | {row['elapsed_seconds']:.3f} |"]
        reached = summary['reached_1_percent_seconds']
        lines += ['', f"到 1%：{f'{reached:.3f} 秒' if reached is not None else '未达到'}；到 0.1%："
            f"{summary['reached_0_1_percent_seconds'] if summary['reached_0_1_percent_seconds'] is not None else '未达到'}。", '',
            '## 分项耗时', '', '| 数值阶段 | 累计秒数 |', '|---|---:|']
        labels = dict(initialization='初始化', relaxation='连续松弛', qubo='QUBO 编码', annealing='SA 采样',
            repair='候选修复', local_dispatch='局部调度', subproblems='全天子问题', validation='物理与费用校验',
            context='LLM 上下文', llm='主线程 LLM 处理', coordination='乘子协调', certification='整数认证')
        for key, seconds in summary['timings'].items():
            lines += [f"| {labels.get(key, key)} | {seconds:.6f} |"]
        lines += ['', f"进程墙钟 {summary['process'].get('wall_seconds', 0):.3f} 秒；"
            f"最后一次预算内保存完成于 {final['elapsed_seconds']:.3f} 秒。"
            '分项为该检查点的累计计时，启动、加载、授权、保存及被中断的工作由总墙钟约束。'
            '异步子代理等待与整数认证重叠，不能将两者相加作为总耗时。', '',
            '## SA 与子代理', '',
            f"SA 原始候选 {summary['sa']['raw_candidates']} 个，原始离散可行 "
            f"{summary['sa']['raw_discrete_feasible']} 个；预算内已处理 {summary['sa']['processed_candidates']} 个。"
            f"其中接受 {summary['sa']['outcomes'].get('accepted', 0)} 个，重复 "
            f"{summary['sa']['outcomes'].get('duplicate', 0)} 个；修复 Hamming 距离范围 {summary['sa']['repair_hamming_range']}。", '',
            f"完整分块轮次 {summary['complete_rounds']}，正式乘子协调 {summary['coordination_count']} 次。"
            '有效上界下降按直接产生方案的来源归属：'+
            '、'.join(f'{source}：{decrease:.6f}' for source, decrease in summary['upper_decrease_by_source'].items())+'。', '',
            f"正式上界请求 {summary['formal_llm_calls_by_channel'].get('upper', 0)} 次，"
            f"下界请求 {summary['formal_llm_calls_by_channel'].get('lower', 0)} 次。", '',
            '| 正式请求 | 通道 | 状态 | 请求至完成确认（秒） | 格式有效 / 数值接受 | 新割 | 直接 UB 改善 |',
            '|---|---|---|---:|---:|---:|---:|']
        for receipt in summary['formal_receipts']:
            seconds = receipt.get('seconds')
            lines += [f"| {receipt['call_id']} | {receipt['channel']} | {receipt['status']} | "
                f"{seconds:.3f}"+f" | {receipt.get('format_valid')} / {receipt.get('accepted')} | "
                f"{receipt.get('new_global_cuts')} | {receipt.get('direct_incumbent_decrease')} |"] if seconds is not None else []
        lines += ['', '代理标识、Astra/max 设置、完整提示、请求及回复 SHA256、时间和逐候选数值反馈，'
            '保存在 [正式回执](run/llm_receipts.json) 及 `agent_bridge/` 对应请求目录。', '']
        if valid and summary.get('final_solution'):
            solution = summary['final_solution']
            lines += ['## 独立审计与最终方案', '',
                f"二次目标的独立初始对偶值 {audited['initial_dual_value']:.6f}；"
                f"{audited['global_cuts']} 条全天割、{audited['hourly_cuts']} 条小时割已核验。"
                f"预算内 {len(audited['incumbents'])} 个不同的 incumbent 全部通过新建固定开机 QP 重算，"
                '上下界单调且 LB ≤ UB。下界采用 Gurobi 主问题有效界和核验过的割。', '',
                f"[最终方案与全部场景调度]({solution['artifact']}) 的固定费用为 {solution['fixed_cost']:.6f}。", '',
                '| 场景 | 概率 | 二次燃料费用 |', '|---|---:|---:|']
            for scenario in solution['scenario_costs']:
                lines += [f"| {scenario['name']} | {scenario['probability']} | {scenario['objective']:.6f} |"]
            lines += ['', f"最终物理校验：{solution['validation']}。独立审计耗时 "
                f"{summary['audit_wall_seconds']:.3f} 秒，另计预算。", '']
        lines += ['## 预检与资源门禁', '', '| 预检阶段 | 墙钟秒数 |', '|---|---:|']
        for name, seconds in summary['preflight_wall_seconds'].items():
            if seconds is not None:
                lines += [f'| {name} | {seconds:.3f} |']
        gate = summary['resource_gate']
        if gate:
            lines += ['', f"正式启动可用内存 {gate['available_bytes']/1024**3:.3f} GiB，"
                f"门槛 {gate['required_bytes']/1024**3:.3f} GiB，已通过。", '']
        lines += [f"核心回归 {summary['core_regression_tests']} 项通过。八个 SA 块、一次协调和重新编码通过；"
            'LLM 预检总预算 1800 秒，上下界各一次真实子代理提案及数值评估通过。'
            '正式运行重新初始化，预检方案与割未传入正式成绩。', '']
    if state.get('error'):
        lines += ['当前阻碍：'+state['error'], '']
    if not summary['formal_llm_calls']:
        lines += ['正式 LLM 调用为零；独立预检中的子代理提案不计入正式成绩。', '']
    lines += ['子代理采用 Astra/max，每次独立上下文，只生成候选开机矩阵。上界建议修复后评估，'
        '下界建议直接检查离散约束并评估；割和下界由数值算法生成。', '',
        '三轮停滞触发，每侧每批最多三次，正式最多 12 次请求且最多一个在途请求；'
        '单次请求上限 600 秒并受剩余总预算约束。3 小时分块，20×20 轮次上限，'
        '整数认证时间片 30 秒，生产 SA 参数见 [冻结选项](options.json)。'
        'gap 为 `(UB-LB)/max(1,abs(UB))`，目标为 0.1%。', '',
        '本次为完整方法单组短测；LLM 的增量贡献需另做对照。SMS100 保持暂停。', '',
        '详细结果见 [summary.json](summary.json)，回执见 [正式回执](run/llm_receipts.json) 和 '
        '[预检回执](preflight/llm/llm_receipts.json)。源码与输入分别冻结在 `snapshot/` 和 `inputs/`。', '']
    if (directory/'progress.png').exists():
        lines += ['![预算内上下界与 gap 轨迹](progress.png)', '', '[PDF 图表](progress.pdf)。', '']
    if 'reference_comparison' in summary:
        reference = summary['reference_comparison']
        pilot = summary['final_solution']
        lines += ['## Gurobi 参考成本对比', '',
            f"相同冻结输入、二次目标及约束，Gurobi 12.0.2、单线程、seed 7，独立冷启动。"
            f"参考求解状态 `{reference['status']}`，耗时 {reference['solve_wall_seconds']:.3f} 秒，"
            f"认证 gap {100*reference['relative_gap']:.6f}%，三个场景重新计算并通过物理与费用校验。", '',
            '| 费用项 | 600 秒完整方法 | Gurobi 参考 | 完整方法减参考 |', '|---|---:|---:|---:|',
            f"| 固定费用 | {pilot['fixed_cost']:.6f} | {reference['fixed_cost']:.6f} | {pilot['fixed_cost']-reference['fixed_cost']:.6f} |",
            f"| 期望燃料费用 | {final['objective']-pilot['fixed_cost']:.6f} | {reference['expected_fuel_cost']:.6f} | {final['objective']-pilot['fixed_cost']-reference['expected_fuel_cost']:.6f} |",
            f"| 总费用 | {final['objective']:.6f} | {reference['objective']:.6f} | {reference['pilot_minus_reference']:.6f} |", '',
            f"相对参考成本差 **{100*reference['pilot_reference_cost_gap']:.6f}%**。"
            '此值与正式 `(UB-LB)/UB` gap 分别报告。参考求解和事后诊断另计时间，未回填正式下界或成绩。', '',
            '[参考方案与三场景调度](reference/result.json)、[Gurobi 原始日志](reference/gurobi.log)。', '']
    if (directory/'analysis.md').exists():
        lines += ['[下界停滞原因与固定割主问题诊断](analysis.md)。', '']
    if (directory/'post_run_changes.json').exists():
        lines += ['报告格式及其回归测试在正式求解和审计完成后补充，数值成绩沿用冻结源码。'
            '差异哈希见 [生成记录](post_run_changes.json)。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(directory, phase):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    if phase == 'report':
        write_report(directory); return 0
    try:
        save(directory/'state.json', dict(status='RUNNING', phase=phase))
        if not (directory/'inputs').exists():
            shutil.copytree(ROOT/'tests/fixtures/rts24', directory/'inputs')
        current = binding(directory)
        save(directory/'environment.json', current['environment'])
        save(directory/'options.json', current['options'])
        run_state = json.loads((directory/'run/state.json').read_text()) if (directory/'run/state.json').exists() else {}
        if not run_state.get('numerical_runs_started'):
            snapshot(directory)
        if phase in ('all', 'preflight'):
            previous = json.loads((directory/'preflight.json').read_text()) if (directory/'preflight.json').exists() else {}
            if previous.get('status') != 'PASSED':
                if run_state.get('numerical_runs_started'):
                    raise ValueError('Cannot rerun preflight after the formal pilot starts')
                run_preflight(directory)
        if phase in ('all', 'run') and not run_state.get('numerical_runs_started'):
            evidence = json.loads((directory/'preflight.json').read_text())
            if evidence['status'] != 'PASSED' or evidence['binding'] != current:
                raise ValueError('Missing, incomplete or stale preflight')
            save(directory/'run/manifest.json', dict(binding=current, time_limit=600, preflight_shared=False))
            pilot_worker('run', directory/'run', 600, options=options(directory),
                         estimated_peak_bytes=evidence['estimated_peak_bytes'], launch_epoch_seconds=time())
        elif phase == 'run':
            raise ValueError('The original 600-second run cannot be repeated or extended')
        if phase in ('all', 'audit') and (directory/'run/trace.jsonl').exists():
            pilot_worker('audit', directory/'audit', 1800,
                         inputs=str(directory/'inputs'), run_directory=str(directory/'run'))
        save(directory/'state.json', dict(status='PREFLIGHT_PASSED' if phase == 'preflight' else 'FINISHED', phase=phase))
        return 0
    except Exception as error:
        state = error.state if isinstance(error, Stopped) else dict(status='FAILED')
        save(directory/'state.json', dict(state, error=str(error), phase=phase))
        pre = directory/'preflight.json'
        if pre.exists():
            evidence = json.loads(pre.read_text())
            if evidence['status'] == 'RUNNING':
                evidence.update(status='FAILED', error=str(error)); save(pre, evidence)
        print(str(error), flush=True)
        return 1
    finally:
        write_report(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('job', type=Path)
    raise SystemExit(worker_main(json.loads(parser.parse_args().job.read_text(encoding='utf-8'))))
