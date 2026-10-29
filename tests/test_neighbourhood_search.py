import numpy as np
import pytest

import neighbourhood_search as ns
from conftest import make_whole
from suc_benders import search as search_core
from suc_benders.anneal import anneal


def whole_and_search(tiny, options, n=3, T=6):
    whole = make_whole(*tiny(n, T), options)
    return whole, ns.NeighbourhoodSearch(whole, options)


def anchor_of(search):
    return np.rint(search.whole.best.point[search.b.cols[search.b.binary]])


def test_parse_neighbourhood_accepts_one_based_input_and_rejects_bad_items(tiny):
    d, _ = tiny(3, 6)
    n = ns.parse_neighbourhood({'units': [3, 1], 'hours': [2, 4], 'reason': 'x'}, d)
    assert (n.units, n.start, n.end) == ((0, 2), 1, 4)
    assert ns.describe(n) == dict(units=[1, 3], hours=[2, 4])
    for bad in ({'units': [], 'hours': [1, 2]}, {'units': [4], 'hours': [1, 2]}, {'units': [1, 1], 'hours': [1, 2]},
                {'units': [1], 'hours': [3, 2]}, {'units': [1], 'hours': [0, 2]}, {'units': [1], 'hours': [1, 7]},
                {'units': [True], 'hours': [1, 2]}, {'units': [1], 'hours': [1, 2], 'extra': 1}, [1, 2]):
        with pytest.raises(ValueError):
            ns.parse_neighbourhood(bad, d)
    with pytest.raises(ValueError):
        ns.parse_neighbourhood({'units': [1, 2, 3], 'hours': [1, 2]}, d, max_units=2)


def test_focus_fixes_every_bit_outside_the_neighbourhood_to_the_incumbent(tiny, options):
    whole, search = whole_and_search(tiny, options)
    n = ns.neighbourhood([0], 1, 4)
    free = search.focus(n)
    anchor = anchor_of(search)
    q, _, _ = search.build_qubo(anchor, [])
    free_u, free_vw = ns.free_masks(whole.d, n)
    position = search.position
    for name, mask in (('u', free_u), ('v', free_vw), ('w', free_vw)):
        outside = getattr(position, name)[~mask]
        assert np.array_equal(q.fixed[outside], anchor[outside])
    assert free == int(np.isnan(search.fixed).sum()) > 0
    assert np.isnan(q.fixed[position.u[free_u]]).any()


def test_samples_change_only_the_neighbourhood(tiny, options):
    whole, search = whole_and_search(tiny, options)
    n = ns.neighbourhood([0, 2], 2, 5)
    search.focus(n)
    q, _, _ = search.build_qubo(anchor_of(search), [])
    candidates, _ = anneal(q, options, 30., [1, 2, 3, 4])
    free_u, _ = ns.free_masks(whole.d, n)
    for x in candidates:
        block_x = np.zeros(len(search.b.obj)); block_x[search.b.binary] = x
        assert np.array_equal(block_x[search.b.u][~free_u], whole.result.u[~free_u])


def test_full_neighbourhood_gives_the_same_qubo_as_a_whole_day_block_search(tiny, options):
    whole, search = whole_and_search(tiny, options)
    search.focus(ns.neighbourhood(range(whole.d.n_units), 0, whole.d.horizon))
    plain = search_core.BlockSearch(whole.full, options, 9, whole)
    anchor = anchor_of(search)
    q1, _, _ = search.build_qubo(anchor, [])
    q2, _, _ = plain.build_qubo(anchor, plain.boundary_rows())
    assert np.array_equal(np.isnan(q1.fixed), np.isnan(q2.fixed))
    assert abs(q1.Q - q2.Q).max() == 0


def test_cuts_persist_across_neighbourhoods_and_candidates_carry_the_neighbourhood_label(tiny, options):
    whole, search = whole_and_search(tiny, options)
    search.focus(ns.neighbourhood([0, 1], 0, 6))
    first = search.step(1)
    stored = len(search.cuts)
    assert first is not None and stored > 0
    search.focus(ns.neighbourhood([2], 0, 6))
    search.step(2)
    assert len(search.cuts) >= stored
    sources = {e['source'] for e in whole.diagnostics['evaluations']}
    assert 'nsa' in sources
    # 全开起点费用最高；至少一个邻域 SA 候选降低了费用，且降幅计在邻域 SA 名下。
    totals = ns.decrease_by_source(whole.diagnostics['evaluations'])
    assert totals['neighbourhood_sa'] > 0 and totals['sa'] == 0


def test_neighbourhood_steps_are_reproducible(tiny, options):
    def run():
        whole, search = whole_and_search(tiny, options)
        for k, n in enumerate((ns.neighbourhood([0, 1], 0, 6), ns.neighbourhood([1, 2], 2, 6)), 1):
            search.focus(n)
            search.step(k)
        return [(e['commitment_hash'], e['objective']) for e in whole.diagnostics['evaluations']]
    assert run() == run()


def test_rule_neighbourhoods_target_committed_units_and_respect_the_size_cap(tiny, options):
    whole, _ = whole_and_search(tiny, options)
    rules = ns.rule_neighbourhoods(whole.d, whole.result, limit=2, max_units=2)
    assert len(rules) == 2
    for n in rules:
        assert 1 <= len(n.units) <= 2 and 0 <= n.start < n.end <= whole.d.horizon


def test_random_moves_change_only_neighbourhood_cells(tiny, options):
    whole, _ = whole_and_search(tiny, options)
    n = ns.neighbourhood([1], 1, 5)
    free_u, _ = ns.free_masks(whole.d, n)
    rng = np.random.default_rng(3)
    for _ in range(20):
        u = ns.random_move(whole.d, whole.result.u, n, rng)
        assert np.array_equal(u[~free_u], whole.result.u[~free_u])


def test_gurobi_rulers_bound_the_incumbent(tiny, options):
    whole, _ = whole_and_search(tiny, options)
    m, full, u = whole.m, whole.full, whole.result.u
    incumbent = whole.result.objective
    unchanged = ns.unit_budget(m, full, u, 0, seconds=30)
    assert unchanged['proven'] and unchanged['changed_units'] == 0
    assert unchanged['objective'] == pytest.approx(incumbent, rel=1e-6)
    ceiling = ns.neighbourhood_ceiling(m, full, u, ns.neighbourhood(range(3), 0, 6), seconds=30)
    budget = ns.unit_budget(m, full, u, 3, seconds=30)
    assert ceiling['proven'] and budget['proven'] and ceiling['dispatch_status'] == 'OPTIMAL'
    assert ceiling['objective'] == pytest.approx(budget['objective'], rel=1e-6)
    assert ns.improves(ceiling['objective'], incumbent)


# 回归：取整后的调度校验不得覆盖 MIP 自己的状态与用时（调度 QP 的 OPTIMAL 不代表 MIP 已证最优）。
def test_ruler_keeps_the_mip_status_after_the_dispatch_check(tiny, options, monkeypatch):
    whole, _ = whole_and_search(tiny, options)
    monkeypatch.setattr(ns, 'evaluate_u', lambda m, full, u: dict(feasible=True, status='OPTIMAL', objective=1.,
                                                                   seconds=.01, commitment_id='x'))
    out = ns.checked(dict(status='TIME_LIMIT', seconds=20., mip_objective=1., bound=.5), whole.m, whole.full,
                     whole.result.u, whole.result.u)
    assert out['status'] == 'TIME_LIMIT' and out['seconds'] == 20. and not out['proven']
    assert out['dispatch_status'] == 'OPTIMAL' and out['check_seconds'] == .01
