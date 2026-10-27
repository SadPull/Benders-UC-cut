"""RTS24 多场景多轮闭环程序训练入口。直接运行创建新归档；之后每轮程序冻结后暂停等待审阅：
在归档 programs/review-round{r}.json 写入 {"P{r}a":{"sha256":"冻结哈希","approved":true,"notes":"审阅结论"},...}，
再以 --resume <归档目录> 继续。每次继续依次完成：审阅 → 本轮探测 → 停止规则 → 未停止时发出下一轮请求。

停止规则、反馈与闭环断言见 program_training.py；至少 4 轮、至多 10 轮。传输、CLI 或超时失败不计轮数，
同一轮最多尝试 3 次，仍失败则暂停（PAUSED_INFRASTRUCTURE），以 --resume 继续同一轮；预算耗尽同样暂停。
LLM 请求、流式采集与 WebSocket 传输移植自已归档的 run_program_synthesis.py。
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import ast
import csv
from datetime import datetime
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
from types import SimpleNamespace as NS
from uuid import uuid4

import numpy as np

import program_evaluation as evaluation
import program_training as lt
import offline_pilot
from offline_worker import check_source
from run_paper_pilot import save, digest, read
from suc_benders.advisor_request import (isolated_home, _prepare_codex, _codex_event_info,
                                        _kill_tree, _remove_home)
from suc_benders.data import jsonable, load_data
from suc_benders.validation import validate, commitment_hash


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'instances/rts24'
# 程序按训练系统与轮次命名：终选程序 L24，第 1 轮最佳程序 L24r1（第 1 轮无有效程序时见
# run_program_evaluation.early_program 的修订，2026-10-26 那次改取第 2 轮的 L24r2）。
FINAL = dict(final=ROOT/'programs/L24.py', early=ROOT/'programs/L24r1.py')
CONTROLLER_FILES = ['offline_pilot.py', 'offline_worker.py', 'program_training.py', 'program_evaluation.py',
                    'run_program_training.py', 'run_paper_pilot.py', 'paper_figures.py', 'requirements.txt', 'programs/L118.py']
STAGE_CAPS = dict(checks=300, anchors=700, controls=1400, rounds=5200)
MODEL, EFFORT = 'gpt-6-astra', 'max'
LLM_SECONDS = 1800.
FORBIDDEN = {'command_execution', 'file_change', 'mcp_tool_call', 'web_search'}
ORIGINAL_INSTRUCTIONS_SHA256 = '1a91ddb3cb484f1fd1c598bfc5018b62416d11043181e3b8f52237c48ddfb7fc'


class BudgetPaused(RuntimeError):
    pass


INSTRUCTIONS = '''Design two small Python UC time-interval proposal functions, not schedules.
Use only this prompt and training feedback. No tools/files/network/external solvers. No imports.
System: {system}.
{objective} Min up/down, initial durations/output,
startup/shutdown capacity limits, and hourly output ramps are enforced. Terminal shutdown is forbidden.
The current feasible u and dispatch p are passed to each function. No repair is applied to its output.
Each accepted candidate undergoes exact continuous dispatch and independent physical/cost validation.
Target: higher time-averaged percentage improvement from a fixed anchor over 15 seconds.
The same function may later be run repeatedly with updated incumbents. Do not use hard-coded unit IDs.

Required source: exactly one top-level def propose(context, rng): ... ; nested helpers are allowed.
No annotations, decorators, classes, exceptions, imports, globals, reflection, or names starting '_'.
context is a read-only dictionary. Arrays are nested tuples. Fields:
n_units, horizon, load[hour], reserve_fraction, u[unit][hour], p[unit][hour], and per-unit vectors
pmin,pmax,pstart,pshut,ramp_up,ramp_down,min_up,min_down,initial_on,initial_duration,initial_output,
cost_a,cost_b,cost_c,startup_cost,shutdown_cost. Initial duration is absolute; initial_on holds state.
Indices in calculations are zero-based; no units or hours are fixed in your source.
Allowed builtins: abs,all,any,bool,dict,enumerate,float,int,len,list,max,min,range,reversed,round,set,sorted,sum,tuple,zip.
Allowed methods: append,extend,copy,sort,index,count,get,items,keys,values.
Only rng has random,randrange,randint,choice,sample,shuffle,uniform methods (Python random.Random).
No other attributes are allowed. Reproducible fresh rng per call; no persistence between calls.
Trusted helper intervals(row) returns zero-based half-open (start,end) pairs for ON runs.
Trusted helper make_candidate(base_u, updates) takes {{zero_based_unit: full_binary_row}} and
returns {{'changes':[{{'unit':1_based_unit,'on_intervals':[{{'start':1_based_inclusive,'end':inclusive}},...]}}]}}.
Return a list of <=4 candidate dicts, each changing 1..4 units. Empty list is valid if no action exists.
Avoid unchanged rows/candidates and duplicates. All candidates in a batch use the SAME input anchor.
Generation hard limit is 0.5 seconds; no evaluator calls or repair. Use efficient bounded loops.
You may design new coordinated interval edits, selection strategies and conditionals. Include diversity
or stochastic sampling so repeated fixed-anchor calls explore different actions. Consider cheap necessary
checks yourself, but do not mistake those checks for actual feasibility or Pmin cost estimates for true gains.
First-hour changes must honor initial status/minimum durations/ramp output; hour-24 shutdown is forbidden.
Min up/down constraints follow the current finite horizon; no additional future-hour commitments are required.
H0 deletes an entire on-run. H1 shifts one start/end by +-1..3 hours. H2 swaps a prefix/suffix/full
on-run window from one unit to another. H3 jointly shortens tails of 2..4 units by 1..3 hours.
Handmade each sample <=64 distinct actions, necessary-check them, then rank removed current per-unit
cost minus added Pmin cost, returning top4. R uniformly draws type/parameters and randomly returns survivors.
Return JSON {{'programs':[{{'reason':'brief mechanism','source':'def propose(context, rng):...'}},...]}}.
Propose two meaningfully different programs. Do not claim numerical gains that have not been measured.'''


# 只描述训练实例本身；IEEE118（单场景、零备用、零启停费）时与原提示逐字一致。
def system_description(d, n_scenarios):
    head = f'aggregated {d.n_units}-unit, {d.horizon}-hour UC, ' + ('one scenario' if n_scenarios == 1 else
                                                                   f'{n_scenarios} load scenarios')
    reserve, start, stop = d.reserve_fraction, bool(np.any(d.startup_cost)), bool(np.any(d.shutdown_cost))
    if reserve == 0 and not start and not stop:
        return head+', zero reserve and startup/shutdown costs'
    parts = ['zero reserve' if reserve == 0 else f'a spinning reserve requirement of {100*reserve:g}% of load',
             ('nonzero' if start else 'zero')+' startup costs and '+('nonzero' if stop else 'zero')+' shutdown costs']
    return head+', '+', '.join(parts)


def objective_description(d):
    extra = ''.join(' plus '+name+' costs' for name, values in (('startup', d.startup_cost), ('shutdown', d.shutdown_cost))
                    if np.any(values))
    return f'Objective includes quadratic fuel a*p^2+b*p plus no-load c*u{extra}.'


def original_instructions(d, n_scenarios=1):
    return INSTRUCTIONS.format(system=system_description(d, n_scenarios), objective=objective_description(d))


# 相对原提示的全部改动：多场景字段与必要检查口径、锚点时刻、轮次说明。每处原句必须恰好出现一次。
SUBSTITUTIONS = [
    ('Target: higher time-averaged percentage improvement from a fixed anchor over 15 seconds.',
     'Target: higher time-averaged percentage improvement over 15 seconds from fixed anchors of a 120 s search\n'
     'with random interval edits and no LLM: after initialization, half-way through its cost reduction, and at its end.'),
    ('n_units, horizon, load[hour], reserve_fraction, u[unit][hour], p[unit][hour], and per-unit vectors',
     'n_units, horizon, load[hour], scenario_loads[scenario][hour], scenario_probabilities[scenario],\n'
     'reserve_fraction, u[unit][hour], p[unit][hour], and per-unit vectors'),
    ('Initial duration is absolute; initial_on holds state.',
     'Initial duration is absolute; initial_on holds state.\n'
     'load and p are probability-weighted over the scenarios. Necessary (not sufficient) checks require, every hour,\n'
     'available capacity of committed units under startup/shutdown limits >= (1+reserve_fraction)*maximum scenario load,\n'
     'summed Pmin <= minimum scenario load, and ramp-reachable output in every scenario.')]


def round_text(round_index):
    head = f'This is round {round_index} of at least {lt.MIN_ROUNDS} and at most {lt.MAX_ROUNDS} rounds.\n'
    if round_index == 1:
        return head+'Later rounds return measured training feedback on every program you propose.'
    return (head+'The input lists every earlier program with its measured training feedback; validation results\n'
            'are withheld. Program 1 must refine the earlier program with the highest train_mean_J, and its reason\n'
            'must state what changed. Program 2 must test a different mechanism aimed at the dominant rejection or\n'
            'duplicate causes shown in the feedback.')


def instructions(d, n_scenarios, round_index):
    text = original_instructions(d, n_scenarios)
    for old, new in SUBSTITUTIONS:
        if text.count(old) != 1:
            raise ValueError('Prompt substitution target not unique: '+old)
        text = text.replace(old, new)
    return text+'\n'+round_text(round_index)


def parse_programs(reply):
    body = json.loads(reply)
    if not isinstance(body, dict) or set(body) != {'programs'} or not isinstance(body['programs'], list) or len(body['programs']) != 2:
        raise ValueError('Exactly two programs are required')
    for row in body['programs']:
        if not isinstance(row, dict) or set(row) != {'source', 'reason'} or not all(isinstance(v, str) for v in row.values()):
            raise ValueError('Programs require string source and reason')
        tree = ast.parse(row['source'])
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != 'propose':
            raise ValueError('Each program must contain only def propose(context, rng)')
        if [a.arg for a in tree.body[0].args.args] != ['context', 'rng']:
            raise ValueError('Required interface propose(context, rng)')
    return body['programs']


# 流式采集：逐行转存 stdout/stderr，实时拦截工具调用，超时整树回收。
def capture(args, env, prompt_path, folder, seconds, verbose=True):
    started = perf_counter()
    messages = queue.Queue()
    threads, process, events = [], None, []
    status = 'FAILED'
    with prompt_path.open('rb') as prompt, (folder/'timeline.jsonl').open('w', encoding='utf-8') as timeline:
        try:
            process = subprocess.Popen(args, stdin=prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, cwd=ROOT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(folder/'running.json', dict(pid=process.pid, timeout_seconds=seconds, status='RUNNING'))

            def pump(stream, name):
                with (folder/name).open('wb', buffering=0) as log:
                    for line in iter(stream.readline, b''):
                        log.write(line)
                        messages.put((name, perf_counter()-started, line))
                stream.close()

            for stream, name in ((process.stdout, 'events.jsonl'), (process.stderr, 'stderr.log')):
                thread = threading.Thread(target=pump, args=(stream, name), daemon=True)
                thread.start()
                threads.append(thread)

            def drain():
                forbidden = False
                while True:
                    try:
                        stream, elapsed, raw = messages.get_nowait()
                    except queue.Empty:
                        return forbidden
                    event = {}
                    if stream == 'events.jsonl':
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            pass
                    item_type = event.get('item', {}).get('type') if isinstance(event, dict) else None
                    record = dict(seconds=elapsed, stream=stream, bytes=len(raw),
                                  event_type=event.get('type') if isinstance(event, dict) else None, item_type=item_type)
                    timeline.write(json.dumps(record)+'\n'); timeline.flush()
                    events.append(record)
                    forbidden |= item_type in FORBIDDEN
                    if verbose:
                        print(f"EVENT {elapsed:.2f}s {stream} {record['event_type'] or ''} {item_type or ''}", flush=True)

            heartbeat = started+60
            while True:
                if drain():
                    status = 'FORBIDDEN_TOOL'
                    break
                code = process.poll()
                if code is not None:
                    status = 'OK' if code == 0 else 'CLI_FAILED'
                    break
                now = perf_counter()
                if now-started >= seconds:
                    status = 'TIME_LIMIT'
                    break
                if now >= heartbeat:
                    if verbose:
                        print(f"WAIT elapsed={now-started:.1f}s last_event={events[-1] if events else None}", flush=True)
                    heartbeat = now+60
                try:
                    process.wait(timeout=min(1., seconds-(now-started)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process is not None and process.poll() is None:
                _kill_tree(process.pid)
                process.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=5)
            if threads:
                if drain():
                    status = 'FORBIDDEN_TOOL'
            ended = perf_counter()-started
    record = dict(status=status, pid=process.pid if process else None,
        exit_code=process.returncode if process else None, seconds=ended, timeout_seconds=seconds,
        first_event_seconds=events[0]['seconds'] if events else None,
        turn_started_seconds=next((r['seconds'] for r in events if r['event_type'] == 'turn.started'), None),
        turn_completed_seconds=next((r['seconds'] for r in events if r['event_type'] == 'turn.completed'), None))
    save(folder/'running.json', record)
    return record


def websocket_args(args):
    """去掉 _prepare_codex 为隔离调用强制加入的 HTTPS 流式提供方设置，改走提供方默认的 WebSocket。"""
    result, k = [], 0
    while k < len(args):
        if args[k] == '-c' and k+1 < len(args) and (args[k+1].startswith('model_provider="suc_https"')
                or args[k+1].startswith('model_providers.suc_https=') or args[k+1].endswith('.supports_websockets=false')):
            k += 2
            continue
        result.append(args[k]); k += 1
    return result


def protocol_text():
    return dict(
        rounds=f'At least {lt.MIN_ROUNDS}, at most {lt.MAX_ROUNDS}; two programs per round',
        completed_round='LLM replied (one format correction allowed) and every approved program finished its probes; '
                        'invalid or rejected programs still complete the round and their reasons enter the next feedback',
        improvement=f'B_r = best validation mean J of all valid programs up to round r; round r improves if '
                    f'B_r >= B_(r-1) + max({lt.DELTA_ABS}, {lt.DELTA_REL}*|B_(r-1)|); round 1 improves if any program is valid',
        stop=f'CONVERGED when r >= {lt.MIN_ROUNDS} and none of the last {lt.PATIENCE} rounds improved; MAX_ROUNDS at r = {lt.MAX_ROUNDS}',
        infrastructure=f'Transport, CLI or timeout failures do not complete a round; at most {lt.ATTEMPTS} attempts per '
                       'invocation, then PAUSED_INFRASTRUCTURE (resumable, never terminal)',
        budget='Sized for the maximum number of rounds; exhaustion pauses (PAUSED_BUDGET) and is never a stop reason',
        feedback=f'Training measurements only (validation withheld) of controls and every earlier program, with summed '
                 f'statistics incl. rejection reasons; full source of the top {lt.SOURCE_TOP} by training mean J plus the previous round',
        closed_loop='Before each request, every earlier program must appear with all training measurements or a recorded '
                    'reason; hashes and run IDs are written to closed-loop.json',
        review='Outcome-blind static review (interface/safety only) before any numerical evaluation',
        selection='final (L24) = validation-best evaluated program (earlier round wins ties); early (L24r1) = validation-best round-1 program',
        gate=f'final program mean J >= {lt.GATE_J} on each validation variant and validation mean above R')


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ('runs', 'anchors', 'programs', 'llm_requests', 'inputs'):
        (output/name).mkdir()
    evaluation.write_instances(output/'test-instances')
    tests = {name: lt.scenario_loads(load_data(output/'test-instances'/name/'units.json',
                                               output/'test-instances'/name/'stochastic.json')[1])
             for name in evaluation.instances()}
    units = read(SOURCE/'units.json')
    variants = lt.variants()
    for name, spec in variants.items():
        folder = output/'inputs'/name
        folder.mkdir()
        save(folder/'units.json', dict(units, base_load=lt.load_profile(units['base_load'], spec['a'], spec['b']).tolist()))
        shutil.copyfile(SOURCE/'stochastic.json', folder/'stochastic.json')
        d, scenarios = load_data(folder/'units.json', folder/'stochastic.json')
        assert d.n_units == 32 and d.horizon == 24 and np.isclose(d.reserve_fraction, .05)
        assert len(scenarios) == 3 and d.startup_cost.all() and not d.shutdown_cost.any()
        spec['test_distance'] = {t: lt.test_distance(lt.scenario_loads(scenarios), loads) for t, loads in tests.items()}
        if min(spec['test_distance'].values()) < lt.MIN_TEST_DISTANCE:
            raise ValueError(f'{name} is too close to a test instance')
    d, scenarios = load_data(output/'inputs/train1/units.json', output/'inputs/train1/stochastic.json')
    first = instructions(d, len(scenarios), 1)
    protocol = dict(title='RTS24 多场景多轮闭环 LLM 区间操作程序训练', variants=variants,
        source_system=str(SOURCE.relative_to(ROOT)),
        parent_input_hashes={p.name: digest(p) for p in (SOURCE/'units.json', SOURCE/'stochastic.json')},
        test_instance_hashes=evaluation.TEST_FILES, minimum_test_distance=lt.MIN_TEST_DISTANCE,
        scenarios='Three source scenarios kept (load factors .9/1/1.05, probabilities .25/.5/.25); base_load reshaped by load_profile',
        numeric_budget_seconds=sum(STAGE_CAPS.values()), allocated_stage_caps=STAGE_CAPS,
        seeds=lt.SEEDS, options=vars(lt.options(17, 60)), controls=lt.CONTROLS,
        anchors=dict(arm=lt.ANCHOR_ARM+' (offline_pilot.run_trial, random interval edits, no LLM)', seconds=lt.ANCHOR_SECONDS,
                     roles=lt.ROLES, half="First incumbent achieving half of the run's total cost reduction",
                     final='Incumbent at the end', reason='SA alone did not improve after initialization in the dry run'),
        probes=dict(seconds=lt.PROBE_SECONDS, cutoff_seconds=lt.PROBE_SECONDS-5, fixed_anchor=True, per_program=lt.counts(),
                    J='Exact mean stepwise percentage improvement over the fixed anchor'),
        generator=dict(max_candidates=4, max_units=4, generation_seconds=.5, no_operator_repair=True,
                       context='offline_pilot.context(..., scenario_fields=True)',
                       handwritten='R/H0-H3 as archived, estimated saving with probability-weighted dispatch'),
        training=protocol_text(),
        llm=dict(model=MODEL, effort=EFFORT, request_seconds=LLM_SECONDS, format_retries=1, transport='websocket',
                 isolated_home=True, forbidden_items=sorted(FORBIDDEN), online_calls_in_search=0,
                 original_instructions_sha256=ORIGINAL_INSTRUCTIONS_SHA256, substitutions=SUBSTITUTIONS,
                 round_text=dict(first=round_text(1), later=round_text(2)),
                 round1_instructions_sha256=hashlib.sha256(first.encode('utf-8')).hexdigest(),
                 withheld='Validation results, test instances, earlier experiments and their diagnostics'),
        evaluation=dict(criteria=evaluation.CRITERIA, limits_seconds=evaluation.LIMITS,
                        jobs=evaluation.jobs(), controls='All control runs repeated in the evaluation experiment'),
        deviations_from_the_two_round_protocol=[
            'Three-scenario training instances; (0,0) dropped because it equals the rts24-base test instance',
            'Programs also receive scenario_loads and scenario_probabilities',
            'Anchors from a 120 s SA+R run (initialization, half of its cost reduction, end); originally a 30 s SA run',
            'Rounds set by the pre-registered stopping rule instead of exactly two',
            'Feedback carries summed statistics with rejection reasons; per-probe timings omitted',
            'Validation probes after every round (needed by the stopping rule); never shown to the LLM'],
        integrity='Cold starts, serial runs, Threads=1; preserve failures; no replacement, retuning or spare-budget reuse')
    save(output/'protocol.json', protocol)
    sources = [ROOT/n for n in CONTROLLER_FILES]
    sources += sorted((ROOT/'suc_benders').glob('*.py')) + sorted((ROOT/'tests').glob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = digest(source)
    inputs = sorted((output/'inputs').rglob('*.json'))+sorted((output/'test-instances').rglob('*.json'))
    manifest = dict(source_files=hashes, input_files={p.relative_to(output).as_posix(): digest(p) for p in inputs},
        protocol_sha256=digest(output/'protocol.json'), python=sys.version, executable=sys.executable,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        frozen_at=datetime.now().astimezone().isoformat())
    save(output/'manifest.json', manifest)
    return protocol, manifest


def verify(output, manifest):
    for relative, expected in manifest['source_files'].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError('Frozen source changed: '+relative)
    for relative, expected in manifest['input_files'].items():
        if digest(output/relative) != expected:
            raise RuntimeError('Frozen input changed: '+relative)
    if digest(output/'protocol.json') != manifest['protocol_sha256']:
        raise RuntimeError('Protocol changed after freeze')


class Experiment:
    def __init__(self, output, resume=False):
        self.output = output
        if resume:
            self.protocol, self.manifest = read(output/'protocol.json'), read(output/'manifest.json')
            self.state = read(output/'state.json')
        else:
            self.protocol, self.manifest = freeze(output)
            self.state = dict(stage='NEW', status='RUNNING', reason='', rows=[], programs={}, allocated={},
                              numeric_seconds=0., rounds=[], closed_loop=[], stop=None, chosen={}, gate={})
        self.check()

    def check(self):
        verify(self.output, self.manifest)
        for identifier, row in self.state['programs'].items():
            if row.get('sha256') and digest(self.output/'programs'/(identifier+'.py')) != row['sha256']:
                raise RuntimeError('Frozen program changed: '+identifier)

    def checkpoint(self):
        save(self.output/'state.json', self.state)

    def reserve(self, stage, seconds):
        self.check()
        allocated = self.state['allocated']
        if (allocated.get(stage, 0)+seconds > self.protocol['allocated_stage_caps'][stage]
                or sum(allocated.values())+seconds > self.protocol['numeric_budget_seconds']):
            raise BudgetPaused(f'Preallocated budget exhausted: {stage}')
        allocated[stage] = allocated.get(stage, 0)+seconds
        self.checkpoint()

    def data(self, variant):
        folder = self.output/'inputs'/variant
        return load_data(folder/'units.json', folder/'stochastic.json')

    def source(self, identifier):
        if identifier in lt.CONTROLS:
            return None
        return (self.output/'programs'/(identifier+'.py')).read_text(encoding='utf-8')

    def done(self, name):
        return next((r for r in self.state['rows'] if r['folder'] == name), None)

    def record(self, name, folder, row, started):
        row['elapsed_seconds'] = perf_counter()-started
        save(folder/'summary.json', row)
        row['elapsed_seconds'] = perf_counter()-started
        self.state['numeric_seconds'] += row['elapsed_seconds']
        self.state['rows'].append(jsonable(row))
        self.checkpoint()
        save(folder/'summary.json', row)
        print(f"DONE {name}: {row['status']} cost={row['objective']:.5f} J={row.get('J')}", flush=True)
        if row['status'] == 'VALIDATION_FAILED':
            raise RuntimeError(row['status'])
        return row

    def folder(self, name):
        folder = self.output/'runs'/name
        if folder.exists():
            raise RuntimeError('Run folder exists without a recorded result; inspect before continuing: '+name)
        folder.mkdir()
        print('RUN '+name, flush=True)
        return folder

    # 无 LLM 的运行：checks 阶段为冒烟，anchors 阶段另存初始化后、成本降幅过半与终点三个锚点。
    def plain(self, stage, variant, seconds):
        name = f'{stage}-{variant}-{lt.ANCHOR_ARM}'
        if self.done(name):
            return self.done(name)
        self.reserve(stage, seconds)
        folder = self.folder(name)
        seed = self.protocol['seeds']['anchors']
        d, scenarios = self.data(variant)
        captures = dict(improvements=[])

        def progress(whole, event):
            t = perf_counter()-whole.started
            if event == 'initialization':
                captures['initial'] = (t, lt.snapshot(whole.result))
            elif np.isfinite(whole.result.objective) and 'initial' in captures and whole.result.objective < (
                    captures['improvements'] or [captures['initial']])[-1][1].objective:
                captures['improvements'].append((t, lt.snapshot(whole.result)))
        started = perf_counter()
        o = lt.options(seed, seconds)
        trial = offline_pilot.run_trial(name, lt.ANCHOR_ARM, seed, seconds, {}, data=(d, scenarios), o=o,
                                        progress_callback=progress)
        anchors = lt.choose_anchors(captures['initial'], captures['improvements']) if 'initial' in captures else {}
        if stage == 'anchors':
            hashes = {k: commitment_hash(v[1]) for k, v in anchors.items()}
            for role, (t, solution) in anchors.items():
                save(self.output/'anchors'/f'{variant}-{role}.json', dict(solution=solution, captured_seconds=t,
                     validation=validate(d, scenarios, solution), duplicate_of=[k for k in lt.ROLES if k != role
                     and hashes.get(k) == hashes[role]], collected_by=name, role=role))
        save(folder/'trial.json', trial)
        row = dict(stage=stage, variant=variant, generator=lt.ANCHOR_ARM, seed=seed, anchor=None, status=trial['status'],
                   message=trial['message'], budget_seconds=seconds, feasible=trial['feasible'], objective=trial['objective'],
                   llm_calls=trial['online_llm_calls'], sa_calls=trial['sa_calls'], folder=name,
                   anchor_objectives={k: v[1].objective for k, v in anchors.items()})
        return self.record(name, folder, row, started)

    def probe(self, budget, stage, variant, identifier, role):
        name = f'{stage}-{variant}-{role}-{identifier}'
        if self.done(name):
            return self.done(name)
        self.reserve(budget, lt.PROBE_SECONDS)
        folder = self.folder(name)
        seed = self.protocol['seeds'][stage]
        d, scenarios = self.data(variant)
        started = perf_counter()
        anchor = lt.restore(read(self.output/'anchors'/f'{variant}-{role}.json')['solution'])
        with (folder/'events.jsonl').open('w', encoding='utf-8') as stream:
            def progress(whole, event):
                row = dict(event=event, seconds=perf_counter()-whole.started, objective=whole.result.objective)
                if event == 'operator':
                    row['operator'] = whole.diagnostics['operator_calls'][-1]
                stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
            result, points = lt.probe(d, scenarios, anchor, identifier, self.source(identifier), seed,
                self.protocol['variants'][variant]['stream_index'], lt.ROLES.index(role), progress, started)
        with gzip.open(folder/'diagnostics.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(jsonable(result.diagnostics), stream, ensure_ascii=False, allow_nan=False)
        row = dict(stage=stage, variant=variant, generator=identifier, seed=seed, anchor=role, status=result.status,
                   message=result.message, budget_seconds=lt.PROBE_SECONDS,
                   feasible=(result.validation or {}).get('feasible', False), objective=result.objective,
                   llm_calls=result.llm_calls, statistics=lt.statistics(result.diagnostics), curve=points,
                   J=lt.improvement_score(points, anchor.objective), folder=name,
                   program_sha256=self.state['programs'].get(identifier, {}).get('sha256'))
        return self.record(name, folder, row, started)

    def probes(self, budget, stage, identifiers):
        names = [v for v in self.protocol['variants'] if v.startswith(stage)]
        for j, variant in enumerate(names):
            for k, role in enumerate(lt.ROLES):
                offset = (len(lt.ROLES)*j+k) % len(identifiers)
                for identifier in identifiers[offset:]+identifiers[:offset]:
                    self.probe(budget, stage, variant, identifier, role)


def request(experiment, folder, body, schema):
    folder.mkdir()
    save(folder/'request.json', body)
    receipt = dict(status='FAILED', configured_model=MODEL, effort=EFFORT, usage=None, cost=None)
    home, reply = None, None
    d, _ = experiment.data('train1')
    try:
        home = isolated_home()
        args, env, receipt['prompt_sha256'] = _prepare_codex('codex', body, NS(n_units=d.n_units, horizon=d.horizon),
                                                             folder, home=home)
        args = websocket_args(args)
        save(folder/'command.json', dict(argv=args, transport='websocket'))
        receipt['transport'] = 'websocket'
        save(folder/'schema.json', schema)
        print(f'LLM {folder.name}: deadline={LLM_SECONDS:.0f}s', flush=True)
        receipt.update(capture(args, env, folder/'prompt.txt', folder, LLM_SECONDS))
        receipt.update(_codex_event_info(folder))
        if receipt['status'] == 'OK':
            reply = (folder/'reply.json').read_text(encoding='utf-8')
            receipt['reply_sha256'] = digest(folder/'reply.json')
    except Exception as error:
        receipt.update(status='FAILED', message=f'{type(error).__name__}: {error}')
    finally:
        if home is not None:
            target = Path(home).resolve()
            if target.parent != Path(tempfile.gettempdir()).resolve() or not target.name.startswith('suc-codex-'):
                raise RuntimeError('Unexpected cleanup path')
            receipt['cleanup'] = _remove_home(target)
        save(folder/'receipt.json', receipt)
    return receipt, reply


def learn(e, r):
    state = e.state
    entries = lt.feedback(state['rows'], state['programs'], r, e.source)
    record = lt.closed_loop_check(r, entries, state['programs'], state['rows'])
    d, scenarios = e.data('train1')
    body = dict(model=MODEL, reasoning=dict(effort=EFFORT), instructions=instructions(d, len(scenarios), r),
        input=json.dumps(jsonable(dict(round=r, training=entries,
            unit_parameters=read(e.output/'inputs/train1/units.json'),
            scenarios=read(e.output/'inputs/train1/stochastic.json'),
            handwritten_source=inspect.getsource(offline_pilot.action)+'\n'+inspect.getsource(lt.handwritten))), ensure_ascii=False))
    item = dict(type='object', additionalProperties=False, required=['reason', 'source'],
                properties={k: dict(type='string') for k in ('reason', 'source')})
    schema = dict(type='object', additionalProperties=False, required=['programs'],
                  properties=dict(programs=dict(type='array', minItems=2, maxItems=2, items=item)))
    state['closed_loop'] = [c for c in state['closed_loop'] if c['round'] != r]+[dict(
        round=r, earlier_programs=record, input_sha256=hashlib.sha256(body['input'].encode('utf-8')).hexdigest(),
        instructions_sha256=hashlib.sha256(body['instructions'].encode('utf-8')).hexdigest())]
    save(e.output/'closed-loop.json', state['closed_loop'])
    e.checkpoint()
    previous = len(list((e.output/'llm_requests').glob(f'round{r}-attempt*')))
    programs, error = None, None
    for attempt in range(previous+1, previous+lt.ATTEMPTS+1):
        receipt, reply = request(e, e.output/'llm_requests'/f'round{r}-attempt{attempt}', body, schema)
        if receipt['status'] in ('CLI_FAILED', 'TIME_LIMIT', 'FAILED'):
            continue
        if receipt['status'] == 'FORBIDDEN_TOOL':
            error = 'FORBIDDEN_TOOL: the reply used a forbidden tool item'
            break
        try:
            programs = parse_programs(reply)
        except (ValueError, SyntaxError, TypeError, KeyError) as first:
            corrected = dict(body, input=body['input']+'\nPrevious reply:\n'+reply+
                             '\nCorrect ONLY JSON/interface syntax, without changing the algorithm: '+str(first))
            receipt, reply = request(e, e.output/'llm_requests'/f'round{r}-attempt{attempt}-format', corrected, schema)
            try:
                if receipt['status'] != 'OK':
                    raise ValueError(receipt['status'])
                programs = parse_programs(reply)
            except (ValueError, SyntaxError, TypeError, KeyError) as second:
                error = f'One permitted format correction exhausted: {first} / {second}'
        break
    else:
        state.update(status='PAUSED_INFRASTRUCTURE',
                     reason=f'Round {r}: {lt.ATTEMPTS} request attempts failed in transport/CLI/timeout; resume continues this round')
        e.checkpoint()
        return False
    earlier = {p['sha256']: i for i, p in state['programs'].items() if p.get('sha256')}
    for k, suffix in enumerate('ab'):
        identifier = f'P{r}{suffix}'
        entry = dict(round=r, status='NO_PROGRAM', error=error, reason=None, sha256=None,
                     frozen_at=datetime.now().astimezone().isoformat())
        if programs is not None:
            path = e.output/'programs'/(identifier+'.py')
            path.write_text(programs[k]['source']+'\n', encoding='utf-8')
            entry.update(sha256=digest(path), reason=programs[k]['reason'], status='PENDING_REVIEW', error=None)
            if entry['sha256'] in earlier:
                entry.update(status='DUPLICATE', error='Identical source to '+earlier[entry['sha256']])
            else:
                try:
                    check_source(programs[k]['source'])
                except (ValueError, SyntaxError) as invalid:
                    entry.update(status='STATIC_INVALID', error=str(invalid))
            earlier.setdefault(entry['sha256'], identifier)
        state['programs'][identifier] = entry
    save(e.output/'programs'/f'freeze-round{r}.json', {i: p for i, p in state['programs'].items() if p['round'] == r})
    state['stage'] = f'REVIEW_ROUND_{r}'
    e.checkpoint()
    print('PROGRAMS FROZEN; REVIEW BEFORE NUMERICAL EVALUATION', flush=True)
    return True


def review(e, r):
    pending = [i for i, p in e.state['programs'].items() if p['round'] == r and p['status'] == 'PENDING_REVIEW']
    if not pending:
        return []
    path = e.output/'programs'/f'review-round{r}.json'
    if not path.exists():
        print('REVIEW REQUIRED '+str(path), flush=True)
        return None
    verdicts, approved = read(path), []
    for identifier in pending:
        program, verdict = e.state['programs'][identifier], verdicts[identifier]
        if verdict['sha256'] != program['sha256']:
            raise RuntimeError('Review hash mismatch: '+identifier)
        program['review'] = verdict
        if verdict['approved']:
            program['status'] = 'APPROVED'
            approved.append(identifier)
        else:
            program.update(status='REVIEW_REJECTED', error='Review: '+verdict['notes'])
    e.checkpoint()
    return approved


def learning_curve(e):
    rows, programs, rounds = e.state['rows'], e.state['programs'], e.state['rounds']
    decision = lt.stop_decision([x['best_validation'] for x in rounds])
    control = lt.mean_j(rows, 'R', 'validation', lt.counts()['validation'])
    table = []
    for k, x in enumerate(rounds):
        mine = [i for i, p in programs.items() if p['round'] == x['round']]
        train = [r for r in rows if r['stage'] == 'train' and r['generator'] in mine]
        table.append(dict(round=x['round'], best_validation=x['best_validation'], improved=decision['improvements'][k],
            programs={i: dict(status=programs[i]['status'], error=programs[i].get('error'),
                              train=lt.mean_j(rows, i, 'train', lt.counts()['train']),
                              validation=lt.mean_j(rows, i, 'validation', lt.counts()['validation'])) for i in sorted(mine)},
            train_statistics=lt._sum_statistics(train) if train else None))
    value = dict(rounds=table, stop=decision, R_validation_mean=control,
                 controls={i: dict(train=lt.mean_j(rows, i, 'train', lt.counts()['train']),
                                   validation=lt.mean_j(rows, i, 'validation', lt.counts()['validation'])) for i in lt.CONTROLS})
    save(e.output/'learning-curve.json', value)
    return value


def figure(e, curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_figures import BLUE, ORANGE, AQUA
    folder = e.output/'figures'; folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for row in curve['rounds']:
        values = [p['validation'] for p in row['programs'].values() if p['validation'] is not None]
        ax.scatter([row['round']]*len(values), values, color=AQUA, s=28, zorder=3,
                   label='Program (validation mean J)' if row['round'] == 1 else None)
    rounds = [r['round'] for r in curve['rounds'] if r['best_validation'] is not None]
    best = [r['best_validation'] for r in curve['rounds'] if r['best_validation'] is not None]
    if rounds:
        ax.step(rounds, best, where='post', color=BLUE, linewidth=2, label='Best so far $B_r$')
    if curve['R_validation_mean'] is not None:
        ax.axhline(curve['R_validation_mean'], color=ORANGE, linestyle='--', linewidth=1.5, label='Random control R')
    ax.set_xlabel('Round'); ax.set_ylabel('Validation mean J (percentage points)')
    ax.set_xticks(range(1, max([r['round'] for r in curve['rounds']]+[1])+1))
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=3)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'F5-learning-curve.{suffix}', dpi=200)
    plt.close(fig)


def finish(e):
    state = e.state
    decision = lt.stop_decision([x['best_validation'] for x in state['rounds']])
    if not decision['stop'] or decision['rounds'] < lt.MIN_ROUNDS:
        raise RuntimeError('Selection reached before the stopping rule allowed it')
    chosen = lt.select(state['rows'], state['programs'])
    state['chosen'] = chosen
    state['gate'] = lt.gate(state['rows'], chosen['final'])
    record = dict(chosen=chosen, gate=state['gate'], stop=decision, programs=state['programs'],
                  frozen_at=datetime.now().astimezone().isoformat(), source_sha256=e.manifest['source_files'])
    if not state['gate']['passed']:
        save(e.output/'final-program-freeze.json', record)
        state.update(status='SCREEN_NOT_PASSED', stage='COMPLETE', reason='终选程序未通过验证门槛；不做验证实验。')
        e.checkpoint()
        return
    for name, path in FINAL.items():
        if path.exists():
            raise RuntimeError('Final program already exists; refusing to overwrite: '+str(path))
    # 第 1 轮没有有效程序时 early 为空，不写 L24r1。
    final = {n: p for n, p in FINAL.items() if chosen[n] is not None}
    for name, path in final.items():
        shutil.copy2(e.output/'programs'/(chosen[name]+'.py'), path)
        assert digest(path) == state['programs'][chosen[name]]['sha256']
    record.update(final_programs={n: dict(path=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),
                                          sha256=digest(p), origin=chosen[n])
                                  for n, p in final.items()}, final_equals_early=chosen['final'] == chosen['early'])
    save(e.output/'final-program-freeze.json', record)
    state.update(status='SELECTED', stage='COMPLETE',
                 reason=f"{decision['reason']} after {decision['rounds']} rounds; final (L24) = {chosen['final']}, early (L24r1) = {chosen['early']}.")
    e.checkpoint()


def report(e):
    output, state = e.output, e.state
    rows = state['rows']
    columns = ['stage', 'variant', 'anchor', 'generator', 'seed', 'status', 'feasible', 'J', 'objective',
               'elapsed_seconds', 'llm_calls', 'folder']
    with (output/'runs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    receipts = [dict(request=p.parent.name, **read(p)) for p in sorted((output/'llm_requests').glob('*/receipt.json'))]
    save(output/'llm-summary.json', dict(calls=len(receipts), seconds=sum(r.get('seconds', 0.) for r in receipts),
                                         cost=None, receipts=receipts))
    curve = learning_curve(e) if state['rounds'] else None
    if curve:
        figure(e, curve)
    fmt = lambda v: '—' if v is None else f'{v:.4f}'
    lines = ['# RTS24 多场景多轮闭环程序训练', '', f"状态：**{state['status']}** / {state['stage']}。{state['reason']}", '',
        f"数值耗时 {state['numeric_seconds']/60:.2f} 分钟；预分配使用 {sum(state['allocated'].values())/60:.2f} / "
        f"{e.protocol['numeric_budget_seconds']/60:.0f} 分钟。",
        f"离线 LLM {len(receipts)} 次请求、{sum(r.get('seconds', 0.) for r in receipts)/60:.1f} 分钟；费用未知，token 见 llm-summary.json。", '']
    if curve:
        lines += [f"停止规则：{curve['stop']['reason']}（完成 {curve['stop']['rounds']} 轮）。R 的验证均分 {fmt(curve['R_validation_mean'])}。", '',
                  '| 轮 | 程序 | 状态 | 训练均分 J | 验证均分 J | B_r | 改进 |', '|---:|---|---|---:|---:|---:|---|']
        for row in curve['rounds']:
            for i, p in row['programs'].items():
                lines.append(f"| {row['round']} | {i} | {p['status']} | {fmt(p['train'])} | {fmt(p['validation'])} | "
                             f"{fmt(row['best_validation'])} | {'是' if row['improved'] else '否'} |")
        lines += ['', '| 对照 | 训练均分 J | 验证均分 J |', '|---|---:|---:|']
        lines += [f"| {i} | {fmt(v['train'])} | {fmt(v['validation'])} |" for i, v in curve['controls'].items()]
    if state['chosen']:
        lines += ['', f"终选：{state['chosen']}。", '', '```json', json.dumps(state['gate'], ensure_ascii=False, indent=2), '```']
    lines += ['', '闭环记录见 closed-loop.json；学习曲线见 learning-curve.json 与 figures/F5-learning-curve。',
              '全部失败和无效程序保留；验证结果从不进入提示。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def continue_experiment(e):
    state = e.state
    if state['status'].startswith('PAUSED'):
        state.update(status='RUNNING', reason='')
    if state['stage'] == 'NEW':
        if not e.done('checks-tests'):
            e.reserve('checks', 120)
            phase = perf_counter()
            with (e.output/'tests.log').open('w', encoding='utf-8') as stream:
                tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=120)
            state['numeric_seconds'] += perf_counter()-phase
            if tests.returncode:
                raise RuntimeError('Tests failed')
            state['rows'].append(dict(stage='checks', generator='tests', folder='checks-tests', status='PASSED'))
            e.checkpoint()
        row = e.plain('checks', 'train1', 30.)
        if row['status'] != 'TIME_LIMIT' or not row['feasible'] or row['llm_calls']:
            raise RuntimeError('Integration smoke failed')
        state['stage'] = 'ANCHORS'; e.checkpoint()
    if state['stage'] == 'ANCHORS':
        for variant in e.protocol['variants']:
            row = e.plain('anchors', variant, lt.ANCHOR_SECONDS)
            if row['status'] != 'TIME_LIMIT' or not row['feasible'] or row['llm_calls']:
                raise RuntimeError('Invalid anchor run; no replacement')
        state['stage'] = 'CONTROLS'; e.checkpoint()
    if state['stage'] == 'CONTROLS':
        for stage in ('train', 'validation'):
            e.probes('controls', stage, lt.CONTROLS)
        state['stage'] = 'LLM_ROUND_1'; e.checkpoint()
    while True:
        stage = state['stage']
        if stage.startswith('LLM_ROUND_'):
            if not learn(e, int(stage.rsplit('_', 1)[1])):
                return
        elif stage.startswith('REVIEW_ROUND_'):
            r = int(stage.rsplit('_', 1)[1])
            approved = review(e, r)
            if approved is None:
                return
            for stage_name in ('train', 'validation'):
                if approved:
                    e.probes('rounds', stage_name, approved)
            for identifier in approved:
                state['programs'][identifier]['status'] = 'EVALUATED'
            best = lt.best_validation(state['rows'], state['programs'], r)
            state['rounds'] = [x for x in state['rounds'] if x['round'] != r]+[dict(
                round=r, best_validation=best, completed_at=datetime.now().astimezone().isoformat())]
            state['stop'] = lt.stop_decision([x['best_validation'] for x in state['rounds']])
            print('ROUND {} COMPLETE: B={} decision={}'.format(r, best, state['stop']['reason']), flush=True)
            state['stage'] = 'SELECT' if state['stop']['stop'] else f'LLM_ROUND_{r+1}'
            e.checkpoint()
        elif stage == 'SELECT':
            finish(e)
            return
        else:
            raise RuntimeError('Experiment is already terminal or stopped; no reruns')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='只继续本次编排；不重跑已完成的运行')
    args = parser.parse_args()
    output = args.resume.resolve() if args.resume else \
        ROOT/'results'/f"program-training-rts24-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    e = Experiment(output, resume=args.resume is not None)
    print('OUTPUT '+str(output), flush=True)
    try:
        continue_experiment(e)
    except BudgetPaused as error:
        e.state.update(status='PAUSED_BUDGET', reason=str(error))
        e.checkpoint()
    except Exception as error:
        e.state.update(status='ABORTED_VALIDITY', stage='STOPPED', reason=f'{type(error).__name__}: {error}')
        e.checkpoint()
        raise
    finally:
        report(e)
        print('STATUS '+e.state['status']+' / '+e.state['stage'], flush=True)


if __name__ == '__main__':
    main()
