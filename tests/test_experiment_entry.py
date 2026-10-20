import gzip
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

import run_experiment as entry
from conftest import whole_day
from suc_benders.anneal import SA_BACKEND, SA_INITIALIZATION
from suc_benders.config import Options
from suc_benders.solver import Result


@pytest.fixture(autouse=True)
def cold_entry(monkeypatch):
    monkeypatch.setattr(entry, 'RESUME_FROM', None)


def test_entry_saves_basic_search_results_and_separate_reference(tmp_path, monkeypatch):
    seen = []
    def solve(options, *, progress, started_at, resume_state):
        seen.append(options)
        assert isinstance(progress, entry.ExperimentProgress)
        assert not hasattr(progress, 'proof_directory') and started_at > 0 and resume_state is None
        return Result(status='TIME_LIMIT', objective=110., lower_bound=100., relative_gap=10./110.,
                      validation={'feasible': True}, diagnostics={'sa_calls': 2, 'block_sa_calls': 2,
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
    assert summary['sa_calls'] == summary['block_sa_calls'] == 2
    assert not {'joint_sa_calls', 'enhanced_search', 'block_sa_period'} & summary.keys()
    assert summary['reference_gap_percent'] == pytest.approx(100*5./105.)
    with (output/'result.json').open(encoding='utf-8') as stream:
        result = json.load(stream)
    assert result['objective'] == 110. and result['lower_bound'] == 100.
    for record in (summary, result):
        assert record['sa_backend'] == SA_BACKEND and record['sa_initialization'] == SA_INITIALIZATION
    assert summary['audit_status'] == 'NOT_REQUESTED'
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
    assert not (tmp_path/'checkpoint.json.gz').exists()
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


def test_resume_entry_adds_budget_preserves_state_and_skips_reference(tmp_path, tiny, options, monkeypatch):
    from test_basic_search import checkpoint
    original = whole_day(tiny, options)
    state = checkpoint(original)
    state['llm_calls'] = 3
    state['llm_history'] = [dict(status='OK', thread_id='prior-session')]
    source = tmp_path/'source'; source.mkdir()
    entry.save(source/'checkpoint.json.gz', state)
    entry.save(source/'manifest.json', dict(files={'data/'+name: hashlib.sha256(
        (entry.PROJECT/'data'/name).read_bytes()).hexdigest() for name in ('units.json', 'stochastic.json')}))
    monkeypatch.setattr(entry, 'RESUME_FROM', str(source/'checkpoint.json.gz'))
    monkeypatch.setattr(entry, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(entry, 'LLM_ENABLED', False)
    monkeypatch.setattr(entry, 'solve_reference', lambda *a: pytest.fail('Reference rerun'))
    def solve(o, *, resume_state, progress, started_at):
        assert o['time_limit'] == state['charged_elapsed_seconds'] + entry.TIME_LIMIT_SECONDS
        assert resume_state == state and progress.prior == state['charged_elapsed_seconds']
        result = original.result
        result.status = 'TIME_LIMIT'; result.elapsed_seconds = progress.prior + 10
        result.llm_calls = 4; progress.session_id = 'prior-session'
        return result
    monkeypatch.setattr(entry, 'run_suc', solve)
    assert entry.main() == 0
    output, = (tmp_path/'results').iterdir()
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['scope'] == 'RESUMED_CONVERGENCE_CHECK'
    assert summary['new_llm_calls'] == 1 and summary['segment_solver_elapsed_seconds'] == 10
    assert not (output/'reference.json').exists()
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['inherited_evidence'] and not manifest['cold_start']
