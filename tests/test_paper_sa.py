from itertools import product
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from audit_helpers import verify_cut
from audit_helpers import quadratic_dual
from suc_benders.benders import dispatch, master
from suc_benders.config import load_options
from suc_benders.model import model, partition
from suc_benders.optimization import continuous_only, optimize
from suc_benders.paper import process_raw_batch, record_state
from suc_benders.qubo import activity_fixed_binary, fixed_binary, qubo
from suc_benders.solver import BlockSearch, Result, WholeDay, solve
from suc_benders.validation import pack


def paper_whole(tiny, options, T=3):
    d, s = tiny(T=T)
    options.paper_sa = True
    m = model(d, s); b = partition(m, T)[0][0]; b.audit_cuts = True
    started = perf_counter()
    w = WholeDay(m, b, options, started+30, Result(), started)
    w.accept(pack(d, np.ones((2, T))), 'initial')
    return w


def test_integer_guard_and_continuous_call_receipts(tiny):
    d, s = tiny(); m = model(d, s); events = []
    with continuous_only(events.append):
        with pytest.raises(ValueError, match='Integer Gurobi'):
            optimize(m, 5, integral=True)
        r = optimize(m, 5)
    assert r.status == 'OPTIMAL'
    assert [e['event'] for e in events] == ['start', 'complete']
    assert all(e['integer_variables'] == 0 and e['threads'] == 1 for e in events)


@pytest.mark.parametrize('overrides', [dict(bound_mode='integer_benders'), dict(llm_enabled=True),
                                     dict(sa_enabled=False), dict(rts24_improvements=True)])
def test_paper_mode_rejects_other_search_backends(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_options(dict(paper_sa=True, **overrides), config_path=tmp_path/'absent')


def test_raw_local_evaluations_precede_all_repairs(tiny, options, monkeypatch):
    w = paper_whole(tiny, options)
    search = BlockSearch(w.full, options, 1, 1, w)
    a = pack(w.d, np.ones((2, 3)))
    b = pack(w.d, np.array([[1, 1, 1], [0, 0, 0]]))
    events = []
    evaluate, prepare = search.evaluate, w.prepare
    def local(x, source):
        events.append(('local', source))
        return evaluate(x, source)
    def repaired(*args):
        assert len([e for e in events if e[0] == 'local']) == 2
        events.append(('repair', ''))
        return prepare(*args)
    monkeypatch.setattr(search, 'evaluate', local)
    monkeypatch.setattr(w, 'prepare', repaired)
    process_raw_batch(search, [a, b, a], w.result.u.copy(), 1)
    assert events[:2] == [('local', 'sa_raw_local')]*2
    assert len(search.cache) == 2 and len(w.diagnostics['paper_candidates']) == 2
    assert {c['source'] for c in w.diagnostics['paper_cuts']} == {'sa_raw_local'}


def test_pricing_context_isolates_cache_and_cuts(tiny, options):
    w = paper_whole(tiny, options, T=6)
    blocks, cross = partition(w.m, 3)
    first = BlockSearch(blocks[0], options, 1, 1, w)
    x = w.best.point[blocks[0].cols[blocks[0].binary]]
    result = first.evaluate(x, 'test')
    assert first.evaluate(x, 'test') is result and len(first.cuts) == 1
    prices = np.zeros(len(w.m.rhs)); prices[cross] = .1
    blocks, _ = partition(w.m, 3, prices)
    second = BlockSearch(blocks[0], options, 2, 1, w)
    assert second.context != first.context and not second.cache and not second.cuts
    assert second.evaluate(x, 'test').feasible
    assert len(w.diagnostics['paper_cuts']) == 2


@pytest.mark.parametrize('rhs', [0., 4.])
def test_activity_propagation_preserves_enumerated_binary_set(rhs, options):
    m = NS(A=sparse.csr_matrix([[1., 1., 1., 1., 0.]]), rhs=np.array([rhs]),
           eq=np.array([True]), binary=np.arange(4), kind=['hard'])
    old = fixed_binary(m); new = activity_fixed_binary(m)
    assert np.isnan(old).all() and not np.isnan(new).any()
    feasible = [x for x in product((0, 1), repeat=4) if sum(x) == rhs]
    reduced = [x for x in product((0, 1), repeat=4) if np.equal(x, new).all()]
    assert feasible == reduced


def test_two_unit_three_hour_cuts_and_quadratic_bounds_enumeration(tiny, options):
    d, _ = tiny(); d.cost_b[1] = 5; d.cost_a[1] = .025; d.ramp_up[0] = 15
    scenarios = [NS(name='low', probability=.25, load=np.array([30., 75., 35.])),
                 NS(name='high', probability=.75, load=np.array([35., 85., 40.]))]
    b = partition(model(d, scenarios), 3)[0][0]
    b.paper_sa = b.audit_cuts = True
    b.constant = 17.25
    cuts, feasible, infeasible = [], [], 0
    for bits in product((0., 1.), repeat=6):
        x = pack(d, np.array(bits).reshape(2, 3))
        r = dispatch(b, x, 5, certified=True)
        if r.cut is not None:
            cuts.append(r.cut)
            verify_cut(b, vars(r.cut))
        if r.feasible:
            feasible.append((x, r))
        else:
            infeasible += 1
    assert feasible and infeasible and any(not c.optimality for c in cuts)
    for c in cuts:
        for x, r in feasible:
            assert c.alpha+c.beta@x <= (r.recourse if c.optimality else 0.)+1e-6
    m, lower = master(b, cuts, 5)
    proof = m.bound_certificate
    prices = np.zeros(len(m.rhs)); prices[proof['indices']] = proof['prices']
    assert lower <= quadratic_dual(m, prices)+b.constant
    assert lower <= min(r.cost for _, r in feasible)+1e-6
    # Fixing implications may not remove any enumerated feasible integer solution.
    fixed = activity_fixed_binary(m)
    for x, _ in feasible:
        known = np.flatnonzero(~np.isnan(fixed[:len(x)]))
        np.testing.assert_array_equal(x[known], fixed[known])
    options.paper_sa = True
    m.lb[-1] = lower-b.constant
    q = qubo(m, options, min(r.cost for _, r in feasible)-b.constant)
    assert q.theta_lower <= lower-b.constant
    assert q.theta_lower+q.theta_weights.sum() >= min(r.cost for _, r in feasible)-b.constant
    assert np.isfinite(q.Q.data).all()


def test_paper_solver_has_only_continuous_calls_and_fresh_start(tiny, options, monkeypatch):
    from suc_benders import solver
    d, s = tiny(); options.paper_sa = True
    options.max_outer, options.max_inner = 2, 2
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([], dict(bits=1)))
    monkeypatch.setattr(solver, 'Certification', lambda *a: pytest.fail('Integer certifier created'))
    r = solve(d, s, options)
    assert not r.status.endswith('FAILED'), r.message
    assert r.objective >= r.lower_bound and r.llm_calls == 0
    assert all(e['integer_variables'] == 0 for e in r.diagnostics['optimizer_calls'])
    assert r.diagnostics['paper_bounds']
    with pytest.raises(ValueError, match='fresh initialization'):
        solve(d, s, options, initial_state={})


def test_paper_budget_reserves_five_seconds(tiny, options):
    d, s = tiny(); options.paper_sa = True; options.time_limit = 4
    r = solve(d, s, options)
    assert r.status == 'TIME_LIMIT' and not r.diagnostics['optimizer_calls']






def test_paper_encoding_preserves_exhaustive_feasible_set(options):
    from suc_benders.paper import decode
    # Duplicate rows and a provably fixed variable must not alter the encoded problem.
    m = NS(A=sparse.csr_matrix([[1., 1., 0., 0.], [0., 0., 1., 0.],
                                [1., 2., 0., -1.], [1., 2., 0., -1.]]),
        rhs=np.array([1., 1., 0., 0.]), eq=np.array([False, True, False, False]),
        obj=np.array([0., 0., 0., 1.]), q=np.zeros(4), lb=np.zeros(4), ub=np.array([1., 1., 1., 3.]),
        binary=np.arange(3), nx=3, and_pairs=[], kind=['hard', 'hard', 'optimality', 'optimality'], scale=10.)
    options.paper_sa = True
    q = qubo(m, options, 3)
    minimum = {}
    for bits in product((0, 1), repeat=q.Q.shape[0]):
        point = tuple(decode(q, ''.join(map(str, bits))))
        energy = float(np.asarray(bits) @ q.Q @ np.asarray(bits)+q.offset)
        minimum[point] = min(minimum.get(point, np.inf), energy)
    assert 3 in q.removed_rows
    for binary in product((0, 1), repeat=3):
        for theta in range(4):
            point = (*binary, theta)
            residual = m.A @ point-m.rhs
            residual[m.eq] = abs(residual[m.eq])
            feasible = max(residual) <= 1e-8
            encoded = abs(minimum.get(point, np.inf)-theta/m.scale) < 1e-8
            assert encoded == feasible
