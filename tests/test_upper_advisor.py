from copy import deepcopy
import json
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from conftest import whole_day
from suc_benders import solver, upper_advisor
from suc_benders.config import load_options


@pytest.fixture
def scheduled(tiny, options, monkeypatch):
    whole = whole_day(tiny, options, n=3)
    options.llm_enabled = True
    requests = []
    def launch(d, o, body, iteration, seconds):
        requests.append(body)
        return NS(poll=lambda:([], dict(status='OK', proposals=[], seconds=0.)))
    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', launch)
    monkeypatch.setattr(whole.certifier, 'step', lambda:None)
    monkeypatch.setattr(whole.certifier, 'converged', lambda:False)
    return whole, requests


def complete(whole, number, sampled=True, allow=True):
    before = whole.diagnostics['sa_calls']
    whole.diagnostics['sa_calls'] += int(sampled)
    whole.advice.after_round(whole.result.objective, before, [], 1, number, allow)


def test_three_complete_sampled_rounds_trigger_only_upper(scheduled):
    whole, requests = scheduled
    for i in (1, 2): complete(whole, i)
    assert not requests
    complete(whole, 3)
    assert len(requests) == 3
    assert all(json.loads(body['input'])['advice_routing']['channel'] == 'upper' for body in requests)
    assert whole.advice.state['phase'] == 'sa'
    assert whole.advice.state['stalled'] == dict(upper=0)


@pytest.mark.parametrize('cap,expected', [(0, 0), (2, 2), (None, 3)])
def test_zero_finite_and_unlimited_caps(scheduled, cap, expected):
    whole, requests = scheduled
    whole.o.llm_max_calls = cap
    for i in range(1, 4): complete(whole, i)
    assert len(requests) == whole.result.llm_calls == expected


def test_unsampled_rounds_and_disabled_tail_do_not_request(scheduled):
    whole, requests = scheduled
    for i in range(1, 4): complete(whole, i, sampled=False)
    assert whole.advice.state['sa_rounds'] == 0
    for i in range(1, 4): complete(whole, i, allow=False)
    assert not requests and whole.advice.state['sa_rounds'] == 3


def test_waiting_keeps_certifying_without_a_second_request(scheduled, monkeypatch):
    whole, requests = scheduled
    whole.o.llm_max_calls = 1
    replies = [None, ([], dict(status='OK', proposals=[], seconds=0.))]
    steps = []
    def launch(*args):
        requests.append(args[2])
        return NS(poll=lambda:replies.pop(0))
    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', launch)
    monkeypatch.setattr(whole.certifier, 'step', lambda:steps.append(True))
    whole.advice.begin()
    whole.advice.run_batch([])
    assert len(requests) == 1 and len(steps) == 2


def test_upper_feedback_refreshes_incumbent_and_all_block_projections(scheduled, monkeypatch):
    whole, requests = scheduled
    whole.o.llm_patience = 1
    searches = [solver.BlockSearch(b, whole.o, 1, j+1, whole)
                for j,b in enumerate(solver.partition(whole.m, 2)[0])]
    def launch(d, o, body, iteration, seconds):
        requests.append(json.loads(body['input']))
        raw = whole.result.u.copy()
        if iteration <= 2: raw[iteration-1] = 0
        record = dict(proposal_id=f'llm:{iteration}:0')
        return NS(poll=lambda:([(raw, record)], dict(status='OK', proposals=[record], seconds=0.)))
    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', launch)
    for i in range(1, 4):
        before = whole.diagnostics['sa_calls']; whole.diagnostics['sa_calls'] += 1
        whole.advice.after_round(whole.result.objective, before, searches, 1, i)
        assert whole.result.llm_calls == i
    assert requests[0]['incumbent_u'][0] == [1, 1, 1]
    assert requests[1]['incumbent_u'][0] == [0, 0, 0]
    assert requests[1]['context']['previous_call_feedback']['proposals'][0]['cost_change']['total'] < 0
    for row in whole.result.llm_history:
        projections = row['proposals'][0]['block_projections']
        assert [p['block'] for p in projections] == [1, 2]
        assert all(p['completed'] and p['feasible'] for p in projections)
    assert all(search.cuts and search.planes for search in searches)
    assert whole.result.llm_history[-1]['proposals'][0]['outcome'] == 'duplicate'
    assert len(whole.advisor_pool) == 2 and whole.result.validation['feasible']


def test_late_reply_never_reaches_upper_evaluation(scheduled):
    whole, _ = scheduled
    whole.advice.begin(); whole.advice.start_request()
    old = whole.result.objective
    whole.deadline = perf_counter()-1.
    assert whole.advice.poll([])
    assert whole.result.objective == old and not whole.advisor_pool
    assert whole.result.llm_history[-1]['status'] == 'TIME_LIMIT'


def test_v029_partial_upper_batch_preserves_spent_calls(scheduled, monkeypatch):
    whole, requests = scheduled
    whole.o.llm_max_calls = None
    whole.advice.begin(outer=1, inner=3)
    batch = whole.advice.state['batch']
    batch.update(last_channel='upper', validation=False)
    batch['counts']['upper'] = 2
    whole.result.llm_calls = 2
    whole.result.llm_history = [dict(status='RUNNING')]
    saved = deepcopy(whole.diagnostics)
    whole.advice.restore(saved)
    assert whole.result.llm_history[0]['status'] == 'CANCELLED'
    whole.advice.run_batch([])
    assert whole.result.llm_calls == 3 and len(requests) == 1


def test_disabled_llm_never_creates_a_request(scheduled):
    whole, requests = scheduled
    whole.o.llm_enabled = False
    for i in range(1, 8): complete(whole, i)
    assert not requests and whole.result.llm_calls == 0


def test_request_honors_long_timeout_and_preserves_validation_budget(scheduled, monkeypatch):
    whole, requests = scheduled
    whole.o.llm_timeout = 1200.
    whole.o.llm_reserve_seconds = 60.
    whole.deadline = perf_counter()+1600.
    budgets = []

    def launch(*args):
        budgets.append(args[-1])
        return NS(poll=lambda: ([], dict(status='OK', proposals=[], seconds=0.)))

    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', launch)
    whole.advice.begin()
    whole.advice.start_request()
    assert budgets == [1200.]
    whole.advice.poll([])
    whole.deadline = perf_counter()+100.
    whole.advice.start_request()
    assert 39. < budgets[-1] <= 40.
    whole.advice.poll([])
    whole.deadline = perf_counter()+59.
    whole.advice.start_request()
    assert len(budgets) == 2 and whole.result.llm_calls == 2
    whole.advice.run_batch([])
    assert whole.diagnostics['llm_batches'][-1]['status'] == 'VALIDATION_RESERVE'


def test_two_failed_calls_finalize_paused_batch(scheduled, monkeypatch):
    whole, requests = scheduled

    def launch(*args):
        return NS(poll=lambda: ([], dict(status='FAILED', proposals=[], seconds=0.)))

    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', launch)
    whole.advice.begin()
    with pytest.raises(upper_advisor.LLMCallFailures):
        whole.advice.run_batch([])
    batch = whole.diagnostics['llm_batches'][-1]
    assert batch['status'] == 'PAUSED_LLM_FAILURE'
    assert batch['ended_seconds'] >= batch['started_seconds']
    assert whole.advice.state['consecutive_failures'] == whole.result.llm_calls == 2
    assert whole.advice.worker is None
    saved = deepcopy(whole.diagnostics)
    whole.advice.restore(saved)
    monkeypatch.setattr(upper_advisor.advisor_request, 'AdvisorRequest', lambda *a:
        NS(poll=lambda: ([], dict(status='OK', proposals=[], seconds=0.))))
    whole.advice.run_batch([])
    assert whole.result.llm_calls == 3 and whole.advice.state['consecutive_failures'] == 0
    assert whole.advice.state['phase'] == 'sa' and whole.advice.state['batch'] is None


@pytest.mark.parametrize('reserve', [-1., float('nan'), True])
def test_invalid_llm_reserve(tmp_path, reserve):
    with pytest.raises(ValueError, match='llm_reserve_seconds'):
        load_options(dict(llm_reserve_seconds=reserve), tmp_path/'absent')


def test_sa_scope_caps_and_charges_dispatch_work(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    options.sa_budget_seconds = 2.
    original = whole.deadline; clock = [perf_counter()]
    monkeypatch.setattr(solver, 'perf_counter', lambda:clock[0])
    with pytest.raises(solver.SABudgetExhausted):
        with whole.sa_work():
            assert whole.remaining() <= 2.
            clock[0] += 2.1
            raise TimeoutError
    assert whole.deadline == original and whole.sa_used() == pytest.approx(2.1)


@pytest.mark.parametrize('key,value', [('bound_mode','strengthened'), ('bound_slice_seconds',1),
                                     ('rts24_improvements',True), ('paper_sa',True)])
def test_removed_options_fail_as_unknown(tmp_path, key, value):
    with pytest.raises(ValueError, match='Unknown option'):
        load_options({key:value}, tmp_path/'absent')


@pytest.mark.parametrize('quota', [-1., float('nan'), True])
def test_invalid_sa_quota(tmp_path, quota):
    with pytest.raises(ValueError, match='SA budget'):
        load_options(dict(sa_budget_seconds=quota), tmp_path/'absent')



def test_zero_sa_quota_still_runs_numerical_proof(tiny, options, monkeypatch):
    options.sa_budget_seconds = 0.
    options.tolerance = .001
    monkeypatch.setattr(solver, 'anneal', lambda *a: pytest.fail('SA quota exhausted'))
    result = solver.solve(*tiny(), options)
    assert result.status == 'CONVERGED', result.message
    assert result.diagnostics['sa_budget_exhausted'] and result.diagnostics['sa_calls'] == 0
    assert result.certification_history and result.validation['feasible']
