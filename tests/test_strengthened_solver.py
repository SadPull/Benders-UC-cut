from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import solver
from suc_benders.config import load_options
from suc_benders.upper_advisor import LLMCallFailures
from suc_benders.model import model, partition
from suc_benders.validation import pack
from conftest import whole_day


def strengthened(tiny, options):
    d, scenarios = tiny()
    m = model(d, scenarios)
    started = perf_counter()
    whole = solver.WholeDay(m, partition(m, d.horizon)[0][0], options,
                           started+30, solver.Result(lower_bound=0.), started)
    whole.accept(pack(d, np.ones((2, d.horizon))), 'initial')
    return whole


def test_new_mode_and_no_imported_initial_state(tiny, tmp_path):
    o = load_options(dict(sa_enabled=False), tmp_path/'absent')
    assert not o.llm_enabled
    with pytest.raises(TypeError, match='initial_state'):
        solver.solve(*tiny(), o, initial_state={})


def test_numeric_worker_cannot_supply_an_upper_candidate(tiny, options, monkeypatch):
    whole = strengthened(tiny, options)
    before = whole.result.objective
    cert = whole.certifier
    rows = [dict(event='finished', status='OPTIMAL', lower=1., point=[0.]*10,
                 elapsed_seconds=1., physical_fingerprint=whole.diagnostics['model_fingerprint'],
                 model_hash='same-fixed-model')]
    cert.connection = NS(poll=lambda:bool(rows), recv=lambda:rows.pop(0))
    cert.process = NS(is_alive=lambda:False)
    monkeypatch.setattr(whole, 'accept', lambda *a, **k:pytest.fail('MIP point reached upper search'))
    cert.poll()
    assert whole.result.objective == before and whole.result.lower_bound == 1.
    assert 'certificate_cuts' not in whole.diagnostics


def test_late_worker_bound_not_scored(tiny, options):
    whole = strengthened(tiny, options)
    cert = whole.certifier
    rows = [dict(event='finished', status='TIME_LIMIT', lower=1., elapsed_seconds=100.,
                 physical_fingerprint=whole.diagnostics['model_fingerprint'], model_hash='model')]
    cert.connection = NS(poll=lambda:bool(rows), recv=lambda:rows.pop(0))
    cert.process = NS(is_alive=lambda:False)
    cert.poll()
    assert whole.result.lower_bound == 0.


def test_two_failed_calls_pause_and_never_request_lower(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    options.llm_enabled, options.llm_max_calls = True, None
    requests = []
    def launch(d, o, body, iteration, seconds):
        requests.append(body)
        return NS(poll=lambda:([], dict(status='FAILED', proposals=[], seconds=0.)))
    monkeypatch.setattr('suc_benders.upper_advisor.advisor_request.AdvisorRequest', launch)
    monkeypatch.setattr(whole.certifier, 'step', lambda:None)
    monkeypatch.setattr(whole.certifier, 'converged', lambda:False)
    whole.advice.begin()
    with pytest.raises(LLMCallFailures):
        for _ in range(3):
            whole.advice.advance([])
    assert len(requests) == 2
    assert all(row['channel'] == 'upper' for row in whole.result.llm_history)
    assert not hasattr(whole.certifier, 'probe_advice')


def test_success_resets_failure_counter_and_restore_keeps_it(tiny, options):
    whole = strengthened(tiny, options)
    advisor = whole.advice
    for status, expected in [('FAILED', 1), ('OK', 0), ('FAILED', 1)]:
        advisor.pending = dict(status=status)
        advisor.record_failure()
        assert advisor.state['consecutive_failures'] == expected
    saved = deepcopy(whole.diagnostics)
    advisor.restore(saved)
    assert advisor.state['consecutive_failures'] == 1
    advisor.pending = dict(status='TIME_LIMIT')
    with pytest.raises(LLMCallFailures):
        advisor.record_failure()


def test_real_cold_proof_and_resume_keep_budget(tiny, options, monkeypatch):
    d, scenarios = tiny(1)
    options.sa_enabled, options.time_limit = False, 15
    result = solver.solve(d, scenarios, options)
    assert result.status == 'CONVERGED', result.message
    assert result.validation['feasible']
    assert 'certificate_cuts' not in result.diagnostics
    assert 'bound_probes' not in result.diagnostics
    assert all(row['source'] == 'initial' for row in result.diagnostics['evaluations'])
    state = deepcopy(vars(result))
    state.update(charged_elapsed_seconds=3., resume_cursor=dict(outer=1, round=1),
                 resume=dict(reason='test'))
    state['diagnostics']['advisor_state']['consecutive_failures'] = 1
    resumed = solver.solve(d, scenarios, options, resume_state=state)
    assert resumed.elapsed_seconds >= 3.
    assert resumed.diagnostics['advisor_state']['consecutive_failures'] == 1
    assert resumed.lower_bound >= result.lower_bound


def test_failed_worker_receipt_invalidates_proof(tiny, options):
    whole = strengthened(tiny, options)
    rows = [dict(event='failed', message='OutOfMemory', elapsed_seconds=1.)]
    whole.certifier.connection = NS(poll=lambda:bool(rows), recv=lambda:rows.pop(0))
    with pytest.raises(RuntimeError, match='OutOfMemory'):
        whole.certifier.poll()


def test_closed_global_gap_skips_further_sampling(tiny, options, monkeypatch):
    backend = solver.StrengthenedCertification
    monkeypatch.setattr(backend,'start',lambda self:None)
    monkeypatch.setattr(backend,'close',lambda self:None)
    monkeypatch.setattr(backend,'step',lambda self:self.update_bound(self.result.objective))
    monkeypatch.setattr(solver,'anneal',lambda *a:pytest.fail('Sampling after certified convergence'))
    result = solver.solve(*tiny(),options)
    assert result.status == 'CONVERGED'
    assert result.diagnostics['complete_rounds'] == 0
