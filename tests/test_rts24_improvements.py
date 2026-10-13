from itertools import product
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from audit_helpers import verify_cut
from suc_benders import solver
from suc_benders.benders import dispatch
from suc_benders.model import model, partition
from suc_benders.scenario_bounds import ScenarioBounds
from suc_benders.validation import pack
from test_certification import whole_day


def test_scenario_cuts_all_commitments_with_ramps(tiny):
    d, _ = tiny()
    d.ramp_up[0], d.cost_b[1], d.cost_a[1] = 15., 5., .02
    scenarios = [NS(name='low', probability=.25, load=np.array([30., 80., 35.])),
                 NS(name='high', probability=.75, load=np.array([35., 85., 40.]))]
    m = model(d, scenarios); full = partition(m, 3)[0][0]
    bounds, feasible, infeasible = ScenarioBounds(full, m), [], 0
    for bits in product([0., 1.], repeat=6):
        x = pack(d, np.array(bits).reshape((2, 3)))
        r = dispatch(full, x, 5, certified=True)
        if r.feasible:
            costs = bounds.recourse(r.point)
            assert costs.sum() == pytest.approx(r.recourse, abs=1e-6)
            bounds.observe(x, r, 'test', 0.)
            feasible.append((x, costs))
        else:
            infeasible += 1
    assert feasible and infeasible and bounds.cuts
    for cut in bounds.cuts:
        verify_cut(bounds.parts[cut['scenario']].b, cut)
        for x, costs in feasible:
            assert cut['alpha']+np.asarray(cut['beta']) @ x <= costs[cut['scenario']]+1e-5
    for x, costs in feasible:
        assert bounds.lower(x) == pytest.approx(sum(costs), abs=1e-3)


def test_adaptive_window_and_constant_floor(tiny, options):
    options.rts24_improvements = True
    whole = whole_day(tiny, options)
    c = whole.certifier
    try:
        c.b.constant = 17.
        c.result.lower_bound = 50.
        c._initialize(); c._sync(); c.g.update()
        assert float(c.variables[-1].LB) == pytest.approx(33.)
        c._advance_window('TIME_LIMIT', 0.)
        assert c.window_index == 0
        c._advance_window('TIME_LIMIT', 0.)
        assert c.window_index == 1
        c._advance_window('OPTIMAL', 0.)
        assert c.window_index == 1 and c.window_stalls == 0
        for _ in range(5):
            c._advance_window('TIME_LIMIT', 0.)
        assert c.window_index == 2
    finally:
        c.close()


def test_improved_solver_closes_gap_and_keeps_native_evidence(tiny, options, monkeypatch):
    d, scenarios = tiny()
    options.rts24_improvements, options.bound_mode = True, 'integer_benders'
    options.max_outer, options.max_inner = 2, 3
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([], {'bits': 1}))
    r = solver.solve(d, scenarios, options)
    assert r.status == 'CONVERGED', r.message
    assert r.objective == pytest.approx(225.)
    assert r.lower_bound <= r.objective and r.relative_gap <= options.tolerance+1e-7
    assert r.diagnostics['native_bounds'] and r.diagnostics['scenario_certificate_cuts']


def test_nonblocking_independent_triggers_and_fairness(tiny, options, monkeypatch):
    from suc_benders import dual_advisor
    options.rts24_improvements, options.llm_enabled, options.llm_max_calls = True, True, 12
    whole = whole_day(tiny, options)
    advisor, cert = whole.advice, whole.certifier
    cert.last_master = dict(x=pack(whole.d, whole.result.u), predicted_cost=100.)
    monkeypatch.setattr(cert, 'converged', lambda: False)
    monkeypatch.setattr(whole, 'upper_context', lambda: ({}, [], None))
    monkeypatch.setattr(dual_advisor.llm, 'request_body', lambda *a: {'input': '{}'})
    monkeypatch.setattr(cert, 'advice_body', lambda *a: {'input': '{}'})
    ready, calls = [False], []
    def launch(d, o, body, iteration, seconds):
        import json
        calls.append(json.loads(body['input'])['advice_routing']['channel'])
        return NS(poll=lambda: ([], dict(status='OK', proposals=[], seconds=1.)) if ready[0] else None)
    monkeypatch.setattr(dual_advisor.bound_advisor, 'BoundAdvisor', launch)
    for k in range(3):
        advisor.after_certificate(whole.result.lower_bound)
    assert advisor.state['stalled']['upper'] == 0
    advisor.service([])
    assert calls == ['lower']
    for k in range(3):
        before = whole.diagnostics['sa_calls']
        whole.diagnostics['sa_calls'] += 1
        advisor.after_round(whole.result.objective, whole.result.lower_bound, before, [], 1, k+1)
    assert calls == ['lower'] and advisor.state['sa_rounds'] == 3
    ready[0] = True
    advisor.service([])
    assert calls == ['lower', 'upper']
    assert advisor.state['stalled']['lower'] == 3


def test_raw_repair_cache_preserves_result(tiny, options, monkeypatch):
    options.rts24_improvements = True
    whole = whole_day(tiny, options)
    original, calls = solver.repair, []
    monkeypatch.setattr(solver, 'repair', lambda *args: (calls.append(1), original(*args))[1])
    a, b = {}, {}
    first = whole.prepare(whole.result.u, 'sa', a)
    second = whole.prepare(whole.result.u, 'sa', b)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1 and b['repair_cache_hit']


def test_callback_deadline_and_crossed_bound(tiny, options, monkeypatch):
    import gurobipy as gp
    options.rts24_improvements = True
    whole = whole_day(tiny, options)
    c = whole.certifier; c.callback_saved = -np.inf
    stopped = []
    g = NS(terminate=lambda: stopped.append(True), cbGet=lambda code: whole.result.objective+1.)
    monkeypatch.setattr(whole, 'remaining', lambda: 1.)
    c._callback(g, gp.GRB.Callback.MIP)
    assert stopped and isinstance(c.callback_error, RuntimeError)
    assert not whole.diagnostics['native_bounds']
    c.callback_error = None; monkeypatch.setattr(whole, 'remaining', lambda: -1.)
    c._callback(g, gp.GRB.Callback.MIP)
    assert len(stopped) == 2 and c.callback_error is None
