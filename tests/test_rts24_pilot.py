import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from benchmarks.rts24 import PilotRecorder, quadratic_dual, write_report
from benchmarks.azema.artifacts import save, sha256


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
    PilotRecorder(tmp_path, 600).relaxation(m, r, 1.)
    proof = json.loads((tmp_path/'initial_bound.json').read_text())
    assert proof['arrays_sha256'] == sha256(tmp_path/'initial_prices.npz')
    assert proof['model_fingerprint'] == fingerprint(m)


def test_no_formal_trace_cannot_be_reported_as_converged(tmp_path):
    save(tmp_path/'state.json', dict(status='PREFLIGHT_PASSED'))
    save(tmp_path/'run/result.json', dict(status='CONVERGED', objective=1., lower_bound=1.))
    write_report(tmp_path)
    result = json.loads((tmp_path/'summary.json').read_text())
    assert not result['target_met'] and result['final'] is None


def test_report_statistics_exclude_uncommitted_and_late_work(tmp_path):
    from benchmarks.azema.artifacts import append
    run = tmp_path/'run'
    save(tmp_path/'state.json', dict(status='FINISHED'))
    checkpoint = dict(u=[[1]], offsets=dict(sampling=1, proposals=1, evaluations=1),
        timings=dict(annealing=.3), complete_rounds=1, coordination_count=0)
    save(run/'checkpoints/001.json', checkpoint)
    final = dict(objective=100., lower_bound=90., relative_gap=.1, elapsed_seconds=590.,
        sa_calls=1, llm_calls=1, checkpoint='checkpoints/001.json',
        checkpoint_sha256=sha256(run/'checkpoints/001.json'))
    append(run/'trace.jsonl', final)
    save(tmp_path/'audit/audit.json', dict(passed=True, final=final))
    append(run/'sampling.jsonl', dict(raw_candidates=16, candidates=16, raw_discrete_feasible=2))
    append(run/'sampling.jsonl', dict(raw_candidates=999, candidates=999, raw_discrete_feasible=999))
    append(run/'proposals.jsonl', dict(origin='sa', outcome='duplicate', hamming_u=3))
    append(run/'proposals.jsonl', dict(origin='sa', outcome='accepted', hamming_u=99))
    append(run/'evaluations.jsonl', dict(source='certificate', incumbent_before=105., incumbent=100.))
    append(run/'evaluations.jsonl', dict(source='llm', incumbent_before=100., incumbent=1.))
    save(run/'llm_receipts.json', [dict(phase='formal', started_seconds=500., consumed_seconds=610.,
        call_id='late', channel='upper', numerical_evaluation_completed=True)])
    write_report(tmp_path)
    result = json.loads((tmp_path/'summary.json').read_text())
    assert result['result_status'] == 'NOT_CONVERGED'
    assert result['sa']['raw_candidates'] == 16 and result['sa']['raw_discrete_feasible'] == 2
    assert result['sa']['outcomes'] == {'duplicate': 1}
    assert result['upper_decrease_by_source'] == {'certificate': 5.}
    assert result['formal_llm_calls_by_channel'] == {'upper': 1}
    assert result['formal_receipts'] == []
