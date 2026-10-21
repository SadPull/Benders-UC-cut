import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import run_suc as api
from conftest import whole_day
from suc_benders import llm, reference, solver, search
from suc_benders.config import load_options


def record(status='OPTIMAL', cost=100., feasible=True):
    return dict(status=status, objective=cost, validation={'feasible': feasible}, elapsed_seconds=0.)


@pytest.mark.parametrize('status,cost,objective,feasible,gap,reached', [
    ('OPTIMAL', 100., 100.09, True, .09, True),
    ('OPTIMAL', 100., 100.2, True, .2, False),
    ('TIME_LIMIT', 100., 99., True, -1., False),
    ('OPTIMAL', 100., 99., True, -1., True),
    ('OPTIMAL', -100., -101., True, -1., True),
    ('OPTIMAL', 0., 1e-6, True, None, True),
    ('OPTIMAL', 0., 1e-3, True, None, False),
    ('TIME_LIMIT', None, 100., True, None, False),
    ('OPTIMAL', 100., 100., False, None, False),
    ('REFERENCE_FAILED', None, 100., False, None, False),
])
def test_reference_gap_and_stopping(status, cost, objective, feasible, gap, reached):
    result = solver.Result(objective=objective, validation={'feasible': True},
                           reference=record(status, cost, feasible))
    assert result.reference_gap_percent == (None if gap is None else pytest.approx(gap))
    assert result.reference_target_reached(.001) == reached
    result.validation = None
    assert result.reference_gap_percent is None and not result.reference_target_reached(.001)


@pytest.mark.parametrize('status,has_point,valid,cost_match', [
    ('OPTIMAL', True, True, True),
    ('TIME_LIMIT', True, True, True),
    ('TIME_LIMIT', False, True, True),
    ('INFEASIBLE', False, True, True),
    ('OPTIMAL', True, False, True),
    ('OPTIMAL', True, True, False),
])
def test_reference_only_exposes_validated_cost(tiny, monkeypatch, status, has_point, valid, cost_match):
    d, scenarios = tiny()
    physical = reference.model(d, scenarios)
    raw = NS(status=status, x=np.zeros(len(physical.obj)) if has_point else None,
             objval=100. if cost_match else 101., lower=95.)
    calls = []
    monkeypatch.setattr(reference, 'optimize', lambda m, seconds, **kw: calls.append((seconds, kw)) or raw)
    monkeypatch.setattr(reference, 'unpack', lambda *a: NS(objective=100., u=[[1]]))
    monkeypatch.setattr(reference, 'validate', lambda *a: {'feasible': valid})
    result = reference.solve_reference(d, scenarios, 7., 'reference.log')
    assert calls == [(7., {'integral': True, 'log_file': 'reference.log'})]
    expected = has_point and valid and cost_match
    assert result['objective'] == (100. if expected else None)
    assert ('u' in result) == expected
    assert result['elapsed_seconds'] >= 0 and result['time_limit_seconds'] == 7.
    assert not {'lower', 'gap_percent'} & result.keys()
    # 求解器自身的界始终记录；间隙只在有已校验方案时给出。
    assert result['gurobi_bound'] == 95.
    assert result.get('gurobi_gap_percent') == (pytest.approx(5.) if expected else None)
    if has_point and not expected:
        assert result['status'] == 'REFERENCE_FAILED' and result['message']


def test_reference_exception_records_reason_and_search_continues(tiny, options, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError('reference unavailable')
    monkeypatch.setattr(reference, 'optimize', fail)
    options.sa_enabled = False
    result = solver.solve(*tiny(), options)
    assert result.status == 'SEARCH_EXHAUSTED' and result.validation['feasible']
    assert result.reference['status'] == 'REFERENCE_FAILED'
    assert 'reference unavailable' in result.reference['message']
    assert result.reference_cost is None and result.reference_gap_percent is None


def test_python_entry_solves_reference_once_before_independent_search(tiny, options, monkeypatch, tmp_path):
    now, calls = [100.], []
    options.sa_enabled = False
    options.reference_time_limit = 7.
    monkeypatch.setattr(api, 'load_options', lambda unused: options)
    monkeypatch.setattr(api, 'load_data', lambda *a: tiny())
    monkeypatch.setattr(solver, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(search, 'perf_counter', lambda: now[0])
    def ref(d, scenarios, seconds, log_file):
        calls.append(('reference', seconds, log_file))
        now[0] += 20.
        # 不可迭代的方案会揭露任何把参考方案当作搜索种子的尝试。
        return dict(record('TIME_LIMIT', 1.), u=object(), v=object(), w=object(), elapsed_seconds=20.)
    monkeypatch.setattr(solver, 'solve_reference', ref)
    initialize = search.WholeDay.initialize
    def initial(whole, relaxation):
        calls.append(('search', whole.started, whole.deadline))
        return initialize(whole, relaxation)
    monkeypatch.setattr(search.WholeDay, 'initialize', initial)
    result = api.run_suc(reference_log_file=tmp_path/'ref.log')
    assert calls == [('reference', 7., tmp_path/'ref.log'), ('search', 120., 145.)]
    assert result.status == 'SEARCH_EXHAUSTED' and result.validation['feasible']
    assert result.elapsed_seconds == 0. and result.total_elapsed_seconds == 20.
    assert not {'strengthened_proof', 'proof_shutdown_error'} & result.diagnostics.keys()


@pytest.mark.parametrize('status,expected', [('OPTIMAL', 'REFERENCE_TARGET_REACHED'),
                                           ('TIME_LIMIT', 'SEARCH_EXHAUSTED')])
def test_only_optimal_reference_can_stop_at_initialization(tiny, options, monkeypatch, status, expected):
    d, scenarios = tiny()
    initial = solver.solve(d, scenarios, NS(**dict(vars(options), sa_enabled=False)))
    monkeypatch.setattr(solver, 'solve_reference', lambda *a: record(status, initial.objective))
    options.max_outer = options.max_inner = 1
    result = solver.solve(d, scenarios, options)
    assert result.status == expected
    assert result.reference_gap_percent <= .001


def test_reference_record_does_not_change_llm_prompt(tiny, options):
    whole = whole_day(tiny, options)
    def body():
        context, bases, _ = whole.upper_context()
        return llm.request_body(whole.d, whole.scenarios, whole.result, options, context, bases, 1)
    before = body()
    whole.result.reference = dict(record('TIME_LIMIT', 923456.789), u=[[0]], v=[[0]], w=[[0]])
    assert body() == before
    assert '923456.789' not in json.dumps(before)


@pytest.mark.parametrize('lane', ['regular', 'escalation'])
def test_exhausted_search_drains_pending_llm_work(tiny, options, monkeypatch, lane):
    whole = whole_day(tiny, options)
    run = solver._Run(options, whole.started, {}, whole.started)
    run.whole, run.result = whole, whole.result
    channel = getattr(whole.advice, lane)
    channel.worker = object()
    polls = []
    def advance(searches):
        polls.append(True)
        if len(polls) == 2:
            channel.worker = None
    monkeypatch.setattr(whole.advice, 'advance', advance)
    monkeypatch.setattr(solver, 'sleep', lambda seconds: None)
    solver._finish_advice(run)
    assert len(polls) == 2 and run.result.status == 'SEARCH_EXHAUSTED'


@pytest.mark.parametrize('value', [0, -1, float('nan'), True])
def test_reference_budget_validation(value, tmp_path):
    with pytest.raises(ValueError, match='reference_time_limit'):
        load_options({'reference_time_limit': value}, config_path=tmp_path/'absent.json')
