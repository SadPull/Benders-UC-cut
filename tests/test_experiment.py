"""Current C support: resource limits, durable checkpoints and bound proofs."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from benchmarks import process
from benchmarks.artifacts import Recorder, append, save, scored_trace, sha256
from benchmarks.paper_audit import verify_cut, verify_trace
from benchmarks.rts24 import INPUTS, RTS24, case, quadratic_dual, read


def checkpoint(folder, seconds=10, lower=90, upper=100):
    row = dict(elapsed_seconds=seconds, lower_bound=lower, objective=upper,
               relative_gap=(upper-lower)/upper, checkpoint='checkpoints/1.json')
    save(folder/row['checkpoint'], dict(row, u=[[1]], model_fingerprint='test'))
    row['checkpoint_sha256'] = sha256(folder/row['checkpoint'])
    append(folder/'trace.jsonl', row)
    return row


def test_resource_gate_prevents_process_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(process, 'memory_gate', lambda *a: dict(passed=False, available_bytes=1, required_bytes=4*1024**3))
    monkeypatch.setattr(process.subprocess, 'Popen', lambda *a, **k: pytest.fail('Process started under blocked gate'))
    state = process.run_process(['never'], tmp_path, 30)
    assert state['status'] == 'BLOCKED_MEMORY' and state['numerical_runs_started'] == 0


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


def test_initial_dual_minimizes_quadratic_terms_over_bounds():
    m = NS(A=sparse.csr_matrix([[0.]]), rhs=np.zeros(1), eq=np.zeros(1, bool),
        obj=np.array([-2.]), q=np.ones(1), lb=np.zeros(1), ub=np.array([3.]))
    assert quadratic_dual(m, [0.]) == -1.
    for invalid in ([-1.], [float('nan')], [0., 0.]):
        with pytest.raises(ValueError):
            quadratic_dual(m, invalid)


def test_saved_initial_certificate_has_array_hash_and_fingerprint(tmp_path, tiny):
    from suc_benders.model import model, fingerprint
    m = model(*tiny(1))
    r = NS(lower=1., prices=np.zeros(len(m.rhs)))
    Recorder(tmp_path, 600).relaxation(m, r, 1.)
    proof = json.loads((tmp_path/'initial_bound.json').read_text())
    assert proof['arrays_sha256'] == sha256(tmp_path/'initial_prices.npz')
    assert proof['model_fingerprint'] == fingerprint(m)


def test_core_rts24_data_matches_independent_reference():
    from benchmarks.rts24_paper_c import RTS24 as paper_case
    from suc_benders.model import fingerprint
    manifest = read(RTS24/'reference/manifest.json')
    reference = read(RTS24/'reference/result.json')
    assert paper_case == RTS24 and RTS24.parent.name == 'data'
    assert manifest['input_sha256'] == {name: sha256(RTS24/'inputs'/name) for name in INPUTS}
    assert manifest['model_fingerprint'] == fingerprint(case(RTS24/'inputs'))
    assert reference['checked']['objective'] == pytest.approx(885066.704430, abs=1e-6)
    assert reference['lower_bound'] <= reference['checked']['objective']


def test_checkpoint_hash_and_budget_are_still_enforced(tmp_path):
    row = checkpoint(tmp_path, seconds=10)
    assert scored_trace(tmp_path, 10) == [row]
    assert scored_trace(tmp_path, 9) == []
    save(tmp_path/row['checkpoint'], dict(u=[[0]]))
    with pytest.raises(ValueError, match='hash'):
        scored_trace(tmp_path, 10)


def test_paper_environment_only_reads_current_python_dependencies(monkeypatch):
    from benchmarks import runtime
    packages = dict(numpy='2.2.6', scipy='1.15.3', gurobipy='12.0.2',
                    kaiwu='1.4.1', psutil='6.0.0', pytest='9.1.1')
    monkeypatch.setattr(runtime.importlib.metadata, 'version', packages.__getitem__)
    monkeypatch.setattr(runtime.gp.gurobi, 'version', lambda: (12, 0, 2))
    value = runtime.environment()
    assert value['packages'] == packages and value['gurobi_native'] == '12.0.2'
    packages['gurobipy'] = '13.0.1'
    with pytest.raises(ValueError, match='12.0.2'):
        runtime.environment()


def test_paper_source_snapshot_is_self_contained(tmp_path):
    from benchmarks import artifacts
    hashes = artifacts.snapshot(tmp_path)
    assert {'run_suc.py', 'run_rts24_paper_c.py', 'benchmarks/runtime.py',
            'benchmarks/rts24.py', 'suc_benders/agent_bridge.py'}.issubset(hashes)
    assert not any(name.startswith('benchmarks/azema/') for name in hashes)
    assert not {'run_benchmark.py', 'run_rts24_pilot.py', 'run_rts24_improved.py'}.intersection(hashes)
    assert not {'config.local.json', '.local/gurobi-runtime.lic'}.intersection(hashes)
    assert all(sha256(tmp_path/'snapshot'/name) == digest for name, digest in hashes.items())


def test_paper_entry_loads_core_inputs_and_rejects_stale_preflight(tmp_path, monkeypatch):
    from benchmarks import rts24_paper_c as experiment
    monkeypatch.setattr(experiment, 'launch', lambda *a, **k: pytest.fail('Unvalidated numerical launch'))
    monkeypatch.setattr(experiment, 'environment', lambda: {'python': 'test'})
    for evidence in ({}, {'status': 'PASSED', 'binding': {'source': 'legacy'}}):
        save(tmp_path/'preflight.json', evidence)
        with pytest.raises(ValueError, match='Preflight missing, incomplete or stale'):
            experiment.execute(tmp_path, 'run')
        assert all(sha256(tmp_path/'inputs'/name) == sha256(RTS24/'inputs'/name) for name in INPUTS)
    options = experiment.options(tmp_path, 7)
    assert options['paper_sa'] and options['sa_enabled'] and not options['llm_enabled']
    assert options['bound_mode'] == 'relaxation' and not options['rts24_improvements']


def test_paper_worker_keeps_result_compact(tmp_path, monkeypatch):
    from time import time
    import run_suc
    from benchmarks import rts24_paper_c as experiment
    result = NS(status='TIME_LIMIT', message='', objective=225., lower_bound=220.,
                relative_gap=5/225, elapsed_seconds=1795., diagnostics=object())
    monkeypatch.setattr(run_suc, 'run_suc', lambda *a, **k: result)
    monkeypatch.setattr(experiment, 'memory_gate', lambda *a: {'passed': True})
    monkeypatch.setattr(experiment.gp.gurobi, 'version', lambda: (12, 0, 2))
    assert experiment.worker_main(dict(action='run', directory=str(tmp_path), options={},
        launch_epoch_seconds=time(), allow_low_memory=False)) == 0
    saved = read(tmp_path/'result.json')
    assert saved['objective'] == 225. and saved['status'] == 'TIME_LIMIT'
    assert 'diagnostics' not in saved
