from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import whole_day
from run_suc import _serializable
from suc_benders import solver
from suc_benders.validation import pack


def saved_state(tiny, options):
    whole = whole_day(tiny, options)
    u = whole.result.u.copy(); u[0] = 0
    assert whole.accept(pack(whole.d, u), 'initial')
    state = _serializable(asdict(whole.result))
    state.update(charged_elapsed_seconds=2., resume_cursor=dict(outer=1, round=2),
                 resume=dict(source="test", block_state_rebuilt=True))
    return state


def test_original_v029_checkpoint_resumes_without_resetting_budget(tiny, options, monkeypatch):
    state = json.loads((Path(__file__).parent/'fixtures/v029-strengthened-checkpoint.json').read_text())
    original = deepcopy(state)
    options.sa_enabled = False
    options.sa_budget_seconds = 0.
    monkeypatch.setattr(solver, 'anneal', lambda *a: pytest.fail('No SA quota'))
    result = solver.solve(*tiny(1), options, resume_state=state)
    assert result.status == 'CONVERGED', result.message
    assert result.elapsed_seconds >= 3.
    assert result.diagnostics['timings']['prior_execution'] == 3.
    assert result.lower_bound >= state['lower_bound']
    assert result.diagnostics['advisor_state']['consecutive_failures'] == 1
    assert result.validation['feasible'] and state == original


@pytest.mark.parametrize('corruption', ['bound', 'incumbent', 'cost', 'profile', 'cuts', 'inherited'])
def test_resume_rejects_invalid_recovered_evidence(tiny, options, corruption):
    state = saved_state(tiny, options)
    if corruption == 'bound':
        state['lower_bound'] = state['objective']+1.
    elif corruption == 'incumbent':
        state['u'] = np.zeros((2, 3)).tolist()
    elif corruption == 'cost':
        state['objective'] += 1.
    elif corruption == 'profile':
        del state['diagnostics']['strengthened_proof']
    elif corruption == 'cuts':
        state['diagnostics']['certificate_cuts'] = [dict(alpha=1.)]
    else:
        state['diagnostics']['inherited_evidence'] = dict(lower_bound=1.)
    result = solver.solve(*tiny(), options, resume_state=state)
    assert result.status == 'RESUME_FAILED'


@pytest.mark.parametrize('prior', [-1., float('nan'), 30., 31.])
def test_resume_cannot_reset_or_extend_exhausted_budget(tiny, options, prior):
    with pytest.raises(ValueError, match='resume budget'):
        solver.solve(*tiny(), options, resume_state=dict(charged_elapsed_seconds=prior))


@pytest.mark.parametrize('change', ['cold_cost', 'cold_hours', 'initial_history', 'scenario', 'missing'])
def test_cold_resume_fingerprint(tiny, options, monkeypatch, change):
    def cold_tiny(n=2):
        d, scenarios = tiny(n)
        d.cold_startup_cost = np.full(n, 11.)
        d.cold_start_hours = np.ones(n)
        return d, scenarios
    state = saved_state(cold_tiny, options)
    d, scenarios = cold_tiny()
    if change == 'cold_cost': d.cold_startup_cost += 1
    elif change == 'cold_hours': d.cold_start_hours += 1
    elif change == 'initial_history': d.initial_duration += 1
    elif change == 'scenario': scenarios[0].name = 'renamed'
    else: del state['diagnostics']['model_fingerprint']
    monkeypatch.setattr(solver.StrengthenedCertification, 'restore',
                        lambda *a: pytest.fail('Mismatched proof must not be loaded'))
    result = solver.solve(d, scenarios, options, resume_state=state)
    assert result.status == 'RESUME_FAILED' and 'fingerprint' in result.message


def test_resume_keeps_spent_sa_quota_and_llm_limit(tiny, options, monkeypatch):
    state = saved_state(tiny, options)
    state['diagnostics']['sa_budget_seconds_used'] = 2.
    state['llm_calls'] = 3
    state['llm_history'] = [dict(status='OK', background_seconds=.1) for _ in range(3)]
    options.sa_budget_seconds = 2.
    options.tolerance = .001
    options.llm_enabled = True
    monkeypatch.setattr(solver, 'anneal', lambda *a: pytest.fail('SA quota already spent'))
    monkeypatch.setattr('suc_benders.advisor_request.AdvisorRequest',
                        lambda *a: pytest.fail('LLM limit already spent'))
    result = solver.solve(*tiny(), options, resume_state=state)
    assert result.status == 'CONVERGED', result.message
    assert result.diagnostics['sa_budget_exhausted']
    assert result.diagnostics['sa_budget_seconds_used'] == 2.
    assert result.llm_calls == 3 and len(result.llm_history) == 3
