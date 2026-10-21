from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import search
from suc_benders.benders import hard_rows
from suc_benders.model import partition
from suc_benders.qubo import search_qubo
from suc_benders.validation import pack
from conftest import make_whole


def block_qubo(whole, options, j, cuts=(), boundary=True):
    blocks, _ = partition(whole.m, 3)
    s = search.BlockSearch(blocks[j], options, j+1, whole)
    b = s.b
    anchor = whole.best.point[b.cols[b.binary]]
    q = search_qubo(s.hard, b, list(cuts), anchor, s.boundary_rows() if boundary else [], options)
    return b, anchor, q


def energy(q, x, slack=None):
    z = np.r_[x[q.free], np.zeros(q.Q.shape[0]-len(q.free)) if slack is None else slack]
    return float(z @ q.Q @ z)


def test_transition_violation_costs_hard_weight_and_capacity_ignores_startup_bits(tiny, options):
    d, scenarios = tiny(2, 6)
    whole = make_whole(d, scenarios, options, np.ones((2, 6)))
    b, anchor, q = block_qubo(whole, options, 1)
    position = {j: k for k, j in enumerate(q.free)}
    startup = [position[j] for j in np.r_[b.v.ravel(), b.w.ravel()] if j in position]
    assert startup
    assert q.Q[startup][:, len(q.free):].nnz == 0
    k = position[b.v[0, 1]]
    flipped = anchor.copy(); flipped[q.free[k]] = 1
    assert energy(q, flipped) - energy(q, anchor) >= options.sa_hard_weight - 1


def test_boundary_rows_carry_cross_block_minimum_up_time(tiny, options):
    d, scenarios = tiny(2, 6)
    d.min_up[0] = 3
    u = np.ones((2, 6)); u[0, :2] = 0
    whole = make_whole(d, scenarios, options, u)
    b, _, with_boundary = block_qubo(whole, options, 1)
    _, _, without = block_qubo(whole, options, 1, boundary=False)
    # Unit 1 starts in hour 3, the last hour of block 1; hours 4-5 of block 2 must stay on.
    assert with_boundary.fixed[b.u[0, 0]] == 1 and with_boundary.fixed[b.u[0, 1]] == 1
    assert np.isnan(without.fixed[b.u[0, 0]])


def test_cost_uses_tightest_optimality_cut_at_anchor(tiny, options):
    d, scenarios = tiny(2, 6)
    whole = make_whole(d, scenarios, options, np.ones((2, 6)))
    blocks, _ = partition(whole.m, 3)
    nx = len(blocks[1].binary)
    anchor = whole.best.point[blocks[1].cols[blocks[1].binary]]
    loose = NS(optimality=True, alpha=-1e3, beta=np.full(nx, 1.))
    tight = NS(optimality=True, alpha=0., beta=np.linspace(-1, 1, nx))
    b, _, q = block_qubo(whole, options, 1, cuts=[loose, tight])
    assert tight.alpha + tight.beta @ anchor > loose.alpha + loose.beta @ anchor
    assert np.allclose(q.cost, b.obj[b.binary] + tight.beta)
    _, _, baseline = block_qubo(whole, options, 1)
    difference = q.Q - baseline.Q
    expected = np.zeros(q.Q.shape[0])
    expected[:len(q.free)] = q.cost[q.free] / q.cost_unit - baseline.cost[q.free] / baseline.cost_unit
    assert np.allclose(difference.diagonal(), expected)
    difference.setdiag(0); difference.eliminate_zeros()
    assert difference.nnz == 0


def test_cost_reference_does_not_attract_bits_without_cuts(tiny, options):
    whole = make_whole(*tiny(2, 6), options, np.ones((2, 6)))
    blocks, _ = partition(whole.m, 3)
    b = blocks[1]
    m = hard_rows(b)
    first = whole.best.point[b.cols[b.binary]]
    a = search_qubo(m, b, [], first, [], options)
    z = search_qubo(m, b, [], 1 - first, [], options)
    assert (a.Q != z.Q).nnz == 0


def test_stale_advice_is_rebased_onto_current_incumbent(tiny, options):
    d, scenarios = tiny(2, 3)
    baseline = np.ones((2, 3))
    whole = make_whole(d, scenarios, options, baseline)
    assert whole.accept(pack(d, np.array([[1, 1, 1], [0, 1, 1]])), "pool")
    before = whole.result.objective
    proposal = np.array([[1, 1, 1], [0, 0, 0]])
    records = whole.rebase([(proposal, {"proposal_id": "llm:1:0"})], baseline)
    assert [r["rebased_from"] for r in records] == ["llm:1:0"] and records[0]["units"] == [2]
    assert whole.result.objective < before and whole.result.u[1].tolist() == [0, 0, 0]
    assert whole.diagnostics["evaluations"][-1]["source"] == "llm_rebase"
    assert whole.advisor_pool and whole.search_statistics()["llm_rebase"]["new_feasible"] == 1
