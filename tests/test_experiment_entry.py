import hashlib
import json
from types import SimpleNamespace as NS

import pytest

import run_experiment as entry
from conftest import whole_day
from suc_benders.anneal import SA_BACKEND, SA_INITIALIZATION
from suc_benders.config import Options
from suc_benders.solver import Result


def test_entry_saves_basic_search_results_and_separate_reference(tmp_path, monkeypatch):
    seen = []
    def solve(options, *, progress, started_at):
        seen.append(options)
        assert isinstance(progress, entry.ExperimentProgress)
        assert started_at > 0
        return Result(status='TIME_LIMIT', objective=110., lower_bound=100., relative_gap=10./110.,
                      validation={'feasible': True}, diagnostics={'sa_calls': 2,
                      'sa_backend': SA_BACKEND, 'sa_initialization': SA_INITIALIZATION})
    monkeypatch.setattr(entry, 'run_suc', solve)
    monkeypatch.setattr(entry, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(entry, 'LLM_ENABLED', False)
    monkeypatch.setattr(entry, 'solve_reference', lambda *a:
        dict(status='OPTIMAL', objective=105., lower=105., validation={'feasible': True},
             gap_percent=0., elapsed_seconds=0.))
    monkeypatch.chdir(tmp_path)
    assert entry.main() == 0
    assert len(seen) == 1
    assert not {'enhanced_search', 'block_sa_period'} & seen[0].keys()
    assert seen[0]['seed'] == entry.SEED and seen[0]['block_hours'] == entry.BLOCK_HOURS == 12
    assert seen[0]['sa_sweeps'] == entry.SA_SWEEPS == 16
    output, = (tmp_path/'results').iterdir()
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['sa_calls'] == 2
    assert not {'joint_sa_calls', 'enhanced_search', 'block_sa_period'} & summary.keys()
    assert summary['reference_gap_percent'] == pytest.approx(100*5./105.)
    with (output/'result.json').open(encoding='utf-8') as stream:
        result = json.load(stream)
    assert result['objective'] == 110. and result['lower_bound'] == 100.
    for record in (summary, result):
        assert record['sa_backend'] == SA_BACKEND and record['sa_initialization'] == SA_INITIALIZATION
    effective = json.loads((output/'effective-options.json').read_text(encoding='utf-8'))
    assert effective['sa_reads'] == 8 and effective['llm_max_calls'] is None
    assert effective['time_limit'] == seen[0]['time_limit']


def test_check_only_never_starts_solver_or_creates_results(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'CHECK_ONLY', True)
    monkeypatch.setattr(entry, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(entry, 'run_suc', lambda *a: pytest.fail('Solver started'))
    monkeypatch.setattr(entry, 'solve_reference', lambda *a: pytest.fail('Reference started'))
    assert entry.main() == 0
    assert not (tmp_path/'results').exists()


def test_snapshot_includes_untracked_solver_and_redacts_credentials(tmp_path):
    options = Options(llm_api_key='private-key')
    entry.freeze_experiment(tmp_path, options)
    manifest = json.loads((tmp_path/'manifest.json').read_text(encoding='utf-8'))
    assert 'suc_benders/search.py' in manifest['files']
    assert 'config.local.json' not in manifest['files']
    assert manifest['sa_backend'] == SA_BACKEND and manifest['sa_initialization'] == SA_INITIALIZATION
    assert manifest['dependencies']['dwave-samplers'] == '1.8.0'
    assert manifest['dependencies']['dimod'] == '0.12.22'
    assert manifest['dependencies']['networkx'] == '3.4.2'
    assert 'kaiwu' not in manifest['dependencies']
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((tmp_path/'snapshot'/name).read_bytes()).hexdigest() == expected
    effective = (tmp_path/'effective-options.json').read_text(encoding='utf-8')
    assert 'private-key' not in effective and 'kaiwu' not in effective
    assert json.loads(effective)['llm_api_key'] == '<redacted>'


def test_snapshot_follows_selected_instance_directory(tmp_path):
    folder = entry.PROJECT/'instances'/'classic10'
    entry.freeze_experiment(tmp_path, Options(data_file=str(folder/'units.json'),
                                              scenario_file=str(folder/'stochastic.json')))
    files = json.loads((tmp_path/'manifest.json').read_text(encoding='utf-8'))['files']
    assert {'instances/classic10/units.json', 'instances/classic10/sources/GA10.uc'} <= files.keys()
    assert not any(name.startswith('data/') for name in files)
    assert files['instances/classic10/units.json'] == hashlib.sha256((folder/'units.json').read_bytes()).hexdigest()


def test_snapshot_works_without_git_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.subprocess, 'run', lambda *a, **k: NS(returncode=128, stdout=''))
    entry.freeze_experiment(tmp_path, Options())
    manifest = json.loads((tmp_path/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['git_head'] is None and manifest['git_status'] is None
    assert 'suc_benders/search.py' in manifest['files']


def test_progress_solution_size_is_independent_of_history(tmp_path, tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    progress = entry.ExperimentProgress(tmp_path)
    now = [whole.started + 2.]
    monkeypatch.setattr(entry, 'perf_counter', lambda: now[0])
    progress(whole, 'initialization')
    path = tmp_path/'latest-solution.json'
    first = path.read_bytes()
    state = json.loads(first)
    assert state['objective'] == whole.result.objective
    assert state['elapsed_seconds'] == 2.
    assert state['format'] == 'SOLUTION_SNAPSHOT_NOT_CHECKPOINT'
    assert state['u'] == whole.result.u.tolist()
    whole.result.history = [object()] * 100000
    whole.diagnostics['sampling'] = [object()] * 100000
    progress(whole, 'round')
    assert path.read_bytes() == first
    progress(whole, 'final')
    assert path.read_bytes() == first
    assert len(first) < 10000
    assert not (tmp_path/'latest-solution.tmp.json').exists()


def test_solution_write_failure_preserves_previous_file(tmp_path, tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    progress = entry.ExperimentProgress(tmp_path)
    progress(whole, 'initialization')
    before = (tmp_path/'latest-solution.json').read_bytes()
    def broken_save(path, value):
        path.write_bytes(b'partial write')
        raise OSError('disk write failed')
    monkeypatch.setattr(entry, 'save', broken_save)
    with pytest.raises(OSError, match='disk write failed'):
        progress(whole, 'final')
    assert (tmp_path/'latest-solution.json').read_bytes() == before


def test_llm_record_keeps_base_and_per_candidate_outcomes():
    changes = [dict(unit=1, on_intervals=[])]
    row = dict(call_id='llm_upper:1', status='OK', message='1 of 3 candidates invalid', accepted=1,
               format_valid=2, invalid=1, base_u=[[1., 1.]], response_output=['{}'],
               proposals=[dict(proposal_id='llm:1:0', raw_changes=changes, raw_u=[[0., 0.]], repaired_u=[[0., 0.]],
                               outcome='accepted', hamming_u=0, raw_constraint_violations=[], incumbent_decrease=5.),
                          dict(proposal_id='llm:1:1', raw_changes=[dict(unit=9, on_intervals=[])],
                               outcome='invalid_candidate', invalid_reason='unit 9 out of range 1..1'),
                          dict(proposal_id='llm:1:2', raw_changes=changes, outcome='infeasible', hamming_u=1)],
               feedback=dict(proposals=[dict(proposal_id='llm:1:0', feasible=True, infeasibility_reason=None,
                                             costs=dict(total=95.), cost_change=dict(total=-5.)),
                                        dict(proposal_id='llm:1:1', feasible=None),
                                        dict(proposal_id='llm:1:2', feasible=False, infeasibility_reason='INFEASIBLE',
                                             reachable_shortfall=[dict(hour=9, shortfall_mw=30.)])]),
               rebase=[dict(proposal_id='llm:1:0:rebase:0', rebased_from='llm:1:0', units=[1], raw_u=[[0., 1.]],
                            outcome='duplicate', hamming_u=0, incumbent_decrease=0.)])
    record = entry.llm_record(row)
    assert record['base_u'] == [[1., 1.]] and (record['format_valid'], record['invalid'], record['accepted']) == (2, 1, 1)
    first, invalid, infeasible = record['candidates']
    assert first['raw_changes'] == changes and first['costs'] == {'total': 95.} and first['incumbent_decrease'] == 5.
    assert 'raw_u' not in first and 'repaired_u' not in first
    assert invalid['outcome'] == 'invalid_candidate' and invalid['invalid_reason'] == 'unit 9 out of range 1..1'
    assert infeasible['feasible'] is False and infeasible['infeasibility_reason'] == 'INFEASIBLE'
    assert infeasible['reachable_shortfall'] == [dict(hour=9, shortfall_mw=30.)] and first['reachable_shortfall'] is None
    assert record['rebase'] == [dict(proposal_id='llm:1:0:rebase:0', rebased_from='llm:1:0', units=[1],
                                     outcome='duplicate', repair_bits=0, incumbent_decrease=0.)]
    json.dumps(record)


def test_progress_records_the_event_row_not_the_last_history_row(tmp_path):
    regular = dict(call_id='llm_upper:1', status='OK', tier='regular')
    history = [regular, dict(call_id='llm_escalation:2', status='RUNNING', tier='escalation')]
    whole = NS(result=NS(objective=1., lower_bound=0., llm_calls=2, status='RUNNING', llm_history=history,
                         outer_history=[]),
               started=0., diagnostics=dict(sa_calls=0), advice=NS(thread_id=None, event_row=regular))
    progress = entry.ExperimentProgress(tmp_path)
    progress.last_upper = 1.
    progress(whole, 'llm_response')
    row = json.loads((tmp_path/'progress.jsonl').read_text(encoding='utf-8'))
    assert (row['llm_call_id'], row['llm_tier'], row['llm_status']) == ('llm_upper:1', 'regular', 'OK')


def test_escalation_requires_http_regular_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(entry, 'LLM_ENABLED', True)
    monkeypatch.setattr(entry, 'LLM_TRANSPORT', 'codex')
    monkeypatch.setattr(entry, 'LLM_ESCALATION_MODEL', 'gpt-6-astra')
    monkeypatch.setattr(entry, 'run_suc', lambda *a, **k: pytest.fail('Solver started'))
    with pytest.raises(ValueError, match='escalation'):
        entry.main()
    assert not (tmp_path/'results').exists()


def test_escalation_options_and_records_reach_results(tmp_path, monkeypatch):
    from suc_benders.config import load_options
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(llm=dict(url='http://relay', api_key='key'))), encoding='utf-8')
    monkeypatch.setattr(entry, 'load_options', lambda options: load_options(options, config))
    seen = []
    rows = [dict(call_id='llm_upper:1', status='OK', tier='regular', transport='http'),
            dict(call_id='llm_escalation:2', status='OK', tier='escalation', triggered_by='llm_upper:1',
                 transport='codex_exec', requested_model='gpt-6-astra', deadline_seconds=900., improved=True,
                 codex_home_cleanup='REMOVED')]
    def solve(options, *, progress, started_at):
        seen.append(options)
        return Result(status='TIME_LIMIT', objective=110., lower_bound=100., relative_gap=10./110.,
                      validation={'feasible': True}, llm_calls=2, llm_history=rows,
                      diagnostics={'sa_calls': 2, 'sa_backend': SA_BACKEND,
                                   'sa_initialization': SA_INITIALIZATION})
    monkeypatch.setattr(entry, 'run_suc', solve)
    monkeypatch.setattr(entry, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(entry, 'LLM_ENABLED', True)
    monkeypatch.setattr(entry, 'LLM_TRANSPORT', 'http')
    monkeypatch.setattr(entry, 'LLM_ESCALATION_MODEL', 'gpt-6-astra')
    monkeypatch.setattr(entry, 'solve_reference', lambda *a:
        dict(status='OPTIMAL', objective=105., lower=105., validation={'feasible': True},
             gap_percent=0., elapsed_seconds=0.))
    assert entry.main() == 0
    output, = (tmp_path/'results').iterdir()
    assert seen[0]['llm_escalation_command'] == 'codex'
    assert seen[0]['llm_escalation_directory'] == str(output/'llm_requests')
    assert not seen[0]['llm_codex_command'] and not seen[0]['llm_agent_directory']
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert (summary['llm_escalation_model'], summary['llm_escalation_calls'], summary['llm_escalation_improved']) == (
        'gpt-6-astra', 1, 1)
    result = json.loads((output/'result.json').read_text(encoding='utf-8'))
    escalation = result['llm_history'][1]
    assert (escalation['tier'], escalation['triggered_by'], escalation['transport'], escalation['requested_model'],
            escalation['codex_home_cleanup']) == ('escalation', 'llm_upper:1', 'codex_exec', 'gpt-6-astra', 'REMOVED')
    assert '"key"' not in (output/'effective-options.json').read_text(encoding='utf-8')
