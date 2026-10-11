"""Budget, isolation and mathematical evidence checks without starting a solver."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from benchmarks.azema import process
from benchmarks.azema.artifacts import append, save, scored_trace, sha256
from benchmarks.azema.audit import verify_cut, verify_trace
from benchmarks.azema.protocol import Protocol, options_for, require_preflight
from benchmarks.azema.report import summarize_arm, write_report


def checkpoint(folder, seconds=10, lower=90, upper=100):
    row = dict(elapsed_seconds=seconds, lower_bound=lower, objective=upper,
               relative_gap=(upper-lower)/upper, checkpoint='checkpoints/1.json')
    save(folder/row['checkpoint'], dict(row, u=[[1]], model_fingerprint='test'))
    row['checkpoint_sha256'] = sha256(folder/row['checkpoint'])
    append(folder/'trace.jsonl', row)
    return row


def test_over_budget_final_result_cannot_replace_scored_checkpoint(tmp_path):
    row = checkpoint(tmp_path/'A')
    save(tmp_path/'A/result.json', dict(objective=95, lower_bound=95, status='CONVERGED', elapsed_seconds=3601))
    summary = summarize_arm(tmp_path, 'A', 3600)
    assert summary['objective'] == 100 and summary['lower_bound'] == 90
    assert not summary['audit_passed'] and summary['reached_0_1_percent_seconds'] is None
    assert summary['status'] == 'PENDING_AUDIT'
    assert scored_trace(tmp_path/'A', 3600) == [row]


def test_checkpoint_hash_and_audit_final_must_match(tmp_path):
    row = checkpoint(tmp_path/'A', lower=99.95)
    save(tmp_path/'audits/A/audit.json', dict(passed=True, final=dict(row, objective=99)))
    assert not summarize_arm(tmp_path, 'A', 3600)['audit_passed']
    save(tmp_path/'audits/A/audit.json', dict(passed=True, final=row))
    assert summarize_arm(tmp_path, 'A', 3600)['status'] == 'CONVERGED'
    save(tmp_path/'A/checkpoints/1.json', dict(u=[[0]]))
    with pytest.raises(ValueError, match='hash'):
        scored_trace(tmp_path/'A', 3600)


def test_resource_gate_prevents_process_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(process, 'memory_gate', lambda *a: dict(passed=False, available_bytes=1, required_bytes=4*1024**3))
    monkeypatch.setattr(process.subprocess, 'Popen', lambda *a, **k: pytest.fail('Process started under blocked gate'))
    state = process.run_process(['never'], tmp_path, 30)
    assert state['status'] == 'BLOCKED_MEMORY' and state['numerical_runs_started'] == 0


def test_protocol_freezes_instance_budget_and_disables_channels(tmp_path):
    Protocol().validate()
    for protocol in (Protocol(seed=8), Protocol(time_limit=7200), Protocol(scenarios=10)):
        with pytest.raises(ValueError):
            protocol.validate()
    b, c = (options_for(arm, tmp_path, Protocol()) for arm in 'BC')
    assert not b['sa_enabled'] and not b['llm_enabled']
    assert c['sa_enabled'] and not c['llm_enabled']
    assert not {'llm_api_key', 'llm_url', 'kaiwu_sdk_code', 'kaiwu_user_id'}.intersection(c)


def test_certificate_only_entry_does_not_initialize_kaiwu(monkeypatch):
    import run_suc
    monkeypatch.setattr(run_suc, 'load_data', lambda *a: ('data', 'scenarios'))
    monkeypatch.setattr(run_suc, 'initialize', lambda *a: pytest.fail('Unexpected Kaiwu initialization'))
    monkeypatch.setattr(run_suc, 'solve', lambda *a: 'solved')
    assert run_suc.run_suc(dict(sa_enabled=False, bound_mode='integer_benders')) == 'solved'


def test_cut_dual_proof_rejects_altered_coefficients():
    b = NS(binary=np.array([0]), continuous=np.array([1]), A=sparse.csr_matrix([[-1., -1.]]),
           rhs=np.array([-1.]), eq=np.array([False]), obj=np.array([0., 1.]),
           q=np.zeros(2), lb=np.zeros(2), ub=np.ones(2))
    cut = dict(optimality=True, alpha=1., beta=[-1.], proof=dict(kind='bounded_lagrangian', indices=[0], prices=[1.]))
    verify_cut(b, cut)
    for changed in (dict(cut, alpha=2.), dict(cut, beta=[1.]),
                    dict(cut, proof=dict(kind='bounded_lagrangian', indices=[0], prices=[-1.]))):
        with pytest.raises(ValueError):
            verify_cut(b, changed)


def test_trace_rejects_crossed_bounds_and_reversed_progress():
    row = dict(elapsed_seconds=1, lower_bound=90, objective=100)
    for changed in (dict(row, lower_bound=101), dict(row, lower_bound=89, elapsed_seconds=2),
                    dict(row, objective=101, elapsed_seconds=2), dict(row, elapsed_seconds=0)):
        with pytest.raises(ValueError):
            verify_trace([row, changed])


def test_stale_or_incomplete_preflight_cannot_start_formal_arm(tmp_path):
    path = tmp_path/'preflight.json'
    save(path, dict(status='PASSED', binding=dict(source='old'), passed_checks=[]))
    with pytest.raises(ValueError):
        require_preflight(path, dict(source='new'))
    with pytest.raises(ValueError):
        require_preflight(path, dict(source='old'))


def test_report_with_no_runs_has_no_invented_bounds(tmp_path):
    save(tmp_path/'state.json', dict(status='BLOCKED_MEMORY', available_bytes=2*1024**3, required_bytes=4*1024**3))
    write_report(tmp_path, Protocol())
    summary = json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    assert all(row['objective'] is None for row in summary['arms'].values())
    assert '2.000 GiB' in (tmp_path/'report.md').read_text(encoding='utf-8')


def test_cross_language_converts_json_schedule_to_array(tmp_path, monkeypatch):
    from benchmarks.azema import preflight
    case = next(preflight.tiny_cases())
    monkeypatch.setattr(preflight, 'tiny_cases', lambda: iter([case]))
    monkeypatch.setattr(preflight, 'model', lambda *a: 'model')
    monkeypatch.setattr(preflight, 'partition', lambda *a: (['full'], []))
    monkeypatch.setattr(preflight, 'pack', lambda d, u: u[:, :])
    monkeypatch.setattr(preflight, 'dispatch', lambda *a, **k: NS(feasible=True, cost=1., point=None))
    monkeypatch.setattr(preflight, 'unpack', lambda *a: None)
    monkeypatch.setattr(preflight, 'validate', lambda *a: {})
    monkeypatch.setattr(preflight, 'optimize', lambda *a, **k: NS(status='OPTIMAL', objval=1.))
    monkeypatch.setattr(preflight, 'author_check', lambda *a, **k: dict(status='OPTIMAL', objective=1.))
    preflight.cross_language(dict(directory=str(tmp_path)))
    assert json.loads((tmp_path/'check.json').read_text(encoding='utf-8'))['passed']


def test_common_json_roundtrip_is_accepted_by_commitment_packing(tmp_path, tiny):
    from benchmarks.azema.worker import read_common
    from suc_benders.validation import pack
    d, _ = tiny(1, 2)
    arrays = tmp_path/'common_arrays.npz'
    np.savez_compressed(arrays, relaxation_prices=np.zeros(3))
    save(tmp_path/'common.json', dict(u=[[1, 0]], arrays_sha256=sha256(arrays)))
    loaded = read_common(tmp_path)
    np.testing.assert_array_equal(pack(d, loaded['u']), [1, 0, 1, 0, 0, 1])
