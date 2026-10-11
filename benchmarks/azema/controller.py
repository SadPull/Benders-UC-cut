"""Sequential preflight, four independent arms and separately timed audits."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from time import time
import xml.etree.ElementTree as ET

from . import ROOT, RUNTIME
from .artifacts import memory_gate, save, sha256, snapshot
from .inputs import export_inputs, julia_command, julia_environment, verify_export, verify_sources
from .process import run_process
from .protocol import binding, environment, options_for, require_preflight
from .report import write_report


class Stopped(RuntimeError):
    def __init__(self, state, stage):
        self.state, self.stage = state, stage
        super().__init__(f"{stage}: {state['status']}")


def require_finished(state, stage):
    if state['status'] != 'FINISHED':
        raise Stopped(state, stage)


def worker(action, directory, seconds, **values):
    job = dict(action=action, directory=str(directory), time_limit=seconds, **values)
    save(directory/'job.json', job)
    command = [sys.executable, '-B', '-m', 'benchmarks.azema.worker', str(directory/'job.json')]
    state = run_process(command, directory, seconds)
    require_finished(state, action)
    return state


def check(directory):
    evidence = json.loads((directory/'check.json').read_text(encoding='utf-8'))
    if not evidence.get('passed'):
        raise ValueError(f'Check did not pass: {directory}')
    return evidence


def preflight(directory, protocol):
    inputs, folder = directory/'inputs', directory/'preflight'
    passed = ['environment']
    if (inputs/'author_export.json').exists():
        exported = json.loads((inputs/'author_export.json').read_text(encoding='utf-8'))
        verify_export(exported, 1, protocol.batch, protocol.scenarios)
        save(inputs/'units.json', exported['units'])
        save(inputs/'scenarios.json', exported['scenarios'])
    else:
        export_inputs(inputs, batch=protocol.batch, scenarios=protocol.scenarios)
    passed.append('input_alignment')
    current = binding(protocol, inputs)
    evidence = dict(status='RUNNING', binding=current, passed_checks=passed, estimated_peak_bytes=0)
    save(directory/'preflight.json', evidence)
    peak = 0

    def completed(name, state):
        nonlocal peak
        peak = max(peak, state.get('peak_rss_bytes', 0))
        passed.append(name)
        evidence.update(estimated_peak_bytes=peak)
        save(directory/'preflight.json', evidence)

    regression = folder/'regression'; regression.mkdir(parents=True, exist_ok=True)
    xml = regression/'pytest.xml'
    state = run_process([sys.executable, '-B', '-m', 'pytest', '-q', '-m', 'not integration',
        '-p', 'no:cacheprovider', f'--junitxml={xml}'], regression, 1800)
    require_finished(state, 'core_regression')
    suites = ET.parse(xml).getroot()
    if any(int(s.get(key, 0)) for s in suites.iter('testsuite') for key in ('failures', 'errors', 'skipped')):
        raise ValueError('Core regression contains failed or skipped checks')
    completed('core_regression', state)
    completed('cut_enumeration', state)
    completed('channels_disabled', state)

    state = worker('crosscheck', folder/'cross_language', 3600)
    check(folder/'cross_language'); completed('cross_language', state)
    common = directory/'common'
    if not (common/'common.json').exists():
        state = worker('common', common, protocol.time_limit,
            data_file=str(inputs/'units.json'), scenario_file=str(inputs/'scenarios.json'))
        initial = json.loads((common/'common.json').read_text(encoding='utf-8'))
        # Include imports, solving, saving and process exit, plus publishing this record.
        initial['charged_elapsed_seconds'] = math.ceil(state['wall_seconds']+1.)
        save(common/'common.json', initial)
    else:
        initial = json.loads((common/'common.json').read_text(encoding='utf-8'))
    if initial['charged_elapsed_seconds'] >= protocol.time_limit:
        raise ValueError('Common initialization exhausted the per-arm budget')
    state = worker('common_audit', folder/'common_audit', 3600,
        inputs=str(inputs), common_directory=str(common))
    check(folder/'common_audit'); completed('common_initialization', state)
    state = worker('sa', folder/'sa', 1800, common_directory=str(common), options=options_for('C', inputs, protocol))
    sampling = check(folder/'sa')
    if sampling['llm_calls'] != 0:
        raise ValueError('C preflight unexpectedly called an LLM')
    peak = max(peak, *(q['dense_working_bytes'] for q in sampling['qubo']))
    completed('real_sa_eight_blocks', state)
    state = worker('llm', folder/'llm', 1800, common_directory=str(common), options=options_for('D', inputs, protocol))
    check(folder/'llm'); completed('real_dual_llm', state)
    evidence['status'] = 'PASSED'
    save(directory/'preflight.json', evidence)
    return evidence


def run_arm(directory, protocol, arm, evidence):
    folder, inputs, common = directory/arm, directory/'inputs', directory/'common'
    if (folder/'state.json').exists():
        old = json.loads((folder/'state.json').read_text(encoding='utf-8'))
        if old.get('numerical_runs_started'):
            raise ValueError(f'Arm {arm} already started; its original one-hour run cannot be overwritten or extended')
    initial = json.loads((common/'common.json').read_text(encoding='utf-8'))
    remaining = protocol.time_limit-initial['charged_elapsed_seconds']
    job = dict(action='run', directory=str(folder), common_directory=str(common), time_limit=protocol.time_limit,
        estimated_peak_bytes=evidence['estimated_peak_bytes'], launch_epoch_seconds=time())
    if arm == 'A':
        job.update(action='baseline', author_root=str(RUNTIME/'author'), base=1, batch=protocol.batch,
            scenarios=protocol.scenarios, seed=protocol.seed, tolerance=protocol.tolerance,
            initial=dict(initial, sha256=sha256(common/'common.json')))
    else:
        job['options'] = options_for(arm, inputs, protocol)
    save(folder/'job.json', job)
    save(folder/'manifest.json', dict(arm=arm, protocol=asdict(protocol), binding=evidence['binding'],
        common_sha256=sha256(common/'common.json'), common_charged_seconds=initial['charged_elapsed_seconds'],
        initialization_source_sha256=sha256(common/'initialization_source_sha256.json'),
        independent_arm=True))
    # Reset the launch origin after writing the job and manifest. File preparation is outside a run.
    job['launch_epoch_seconds'] = time()
    save(folder/'job.json', job)
    command = julia_command(folder/'job.json') if arm == 'A' else [sys.executable, '-B', '-m',
        'benchmarks.azema.worker', str(folder/'job.json')]
    state = run_process(command, folder, remaining, env=julia_environment() if arm == 'A' else None,
                        estimated_peak_bytes=evidence['estimated_peak_bytes'])
    if state['status'] == 'BLOCKED_MEMORY':
        raise Stopped(state, arm)
    return state


def execute(directory, protocol, phase, arm):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if phase == 'report':
        write_report(directory, protocol)
        return 0
    arms = tuple('ABCD') if arm == 'ALL' else (arm,)
    stage = 'environment'
    try:
        save(directory/'protocol.json', asdict(protocol))
        save(directory/'state.json', dict(status='RUNNING', phase=phase, stage='environment'))
        save(directory/'environment.json', environment())
        save(directory/'upstream.json', verify_sources())
        if not any((directory/a/'trace.jsonl').exists() for a in 'ABCD'):
            snapshot(directory)
        gate = memory_gate()
        save(directory/'resource_gate.json', gate)
        if not gate['passed']:
            raise Stopped(dict(status='BLOCKED_MEMORY', numerical_runs_started=0, llm_calls_started=0, **gate), 'preflight')
        if phase in ('all', 'preflight'):
            stage = 'preflight'
            save(directory/'state.json', dict(status='RUNNING', phase=phase, stage=stage))
            previous = json.loads((directory/'preflight.json').read_text(encoding='utf-8')) if (directory/'preflight.json').exists() else {}
            if previous.get('status') != 'PASSED':
                if any((directory/a/'trace.jsonl').exists() for a in 'ABCD'):
                    raise ValueError('Cannot regenerate preflight or common initialization after a formal arm starts')
                preflight(directory, protocol)
        evidence = require_preflight(directory/'preflight.json', binding(protocol, directory/'inputs'))
        for a in arms if phase in ('all', 'run', 'audit') else ():
            stage = a
            if phase in ('all', 'run'):
                # Completed arms are retained when continuing an interrupted four-arm sequence.
                state_path = directory/a/'state.json'
                previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
                if not previous.get('numerical_runs_started'):
                    run_arm(directory, protocol, a, evidence)
            if phase in ('all', 'audit') and (directory/a/'trace.jsonl').exists():
                stage = f'audit_{a}'
                worker('audit', directory/'audits'/a, 7200, arm=a, inputs=str(directory/'inputs'),
                    run_directory=str(directory/a), common_directory=str(directory/'common'), budget=protocol.time_limit)
        save(directory/'state.json', dict(status='PREFLIGHT_PASSED' if phase == 'preflight' else 'FINISHED', phase=phase))
        return 0
    except Stopped as error:
        save(directory/'state.json', dict(error.state, stage=error.stage, phase=phase))
        print(str(error), flush=True)
        return 3 if error.state['status'] == 'BLOCKED_MEMORY' else 1
    except Exception as error:
        save(directory/'state.json', dict(status='FAILED', stage=stage, phase=phase, error=f'{type(error).__name__}: {error}'))
        print(f'{stage}: {type(error).__name__}: {error}', flush=True)
        return 1
    finally:
        write_report(directory, protocol)
