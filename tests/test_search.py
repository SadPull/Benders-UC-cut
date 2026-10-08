from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import solver
from suc_benders.model import model, partition
from suc_benders.validation import pack


def make_whole(d, scenarios, options):
    m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
    started = perf_counter()
    whole = solver.WholeDay(m, full, options, started+30, solver.Result(), started)
    assert whole.accept(pack(d, np.ones((d.n_units, d.horizon))), "initial")
    return whole


def force_open_master(monkeypatch):
    original = solver.master
    monkeypatch.setattr(solver, "master", lambda b, cuts, seconds: (original(b, cuts)[0], -1e6))


def test_late_dispatch_not_accepted(tiny, options, monkeypatch):
    d, s = tiny(1); whole = make_whole(d, s, options); b = whole.full
    now = [0.]
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 11.
        return NS(feasible=True, cost=1.)

    monkeypatch.setattr(solver, "dispatch", late)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
    assert result.best is None and not result.planes and result.upper == np.inf


def test_late_relaxation_not_accepted(tiny, options, monkeypatch):
    d, s = tiny(1); whole = make_whole(d, s, options); b = whole.full
    now = [0.]
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 11.
        return None, 225.

    monkeypatch.setattr(solver, "master", late)
    initial_bound = float(np.minimum(b.obj*b.lb, b.obj*b.ub).sum()+b.constant)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
    assert result.lower == initial_bound and result.best is not None


@pytest.mark.parametrize("max_calls,expected", [(2, 2), (0, 0), (None, 9)])
def test_llm_failure_burst_and_fixed_sa_seed(tiny, options, monkeypatch, max_calls, expected):
    d, s = tiny(1); whole = make_whole(d, s, options)
    options.llm_enabled = True; options.llm_patience = 1; options.llm_max_calls = max_calls; options.max_inner = 3
    force_open_master(monkeypatch)
    seeds, requests = [], []

    def sample(q, o, seconds, seed):
        seeds.append(seed)
        return [], {"bits": 1}

    def request(*args):
        requests.append(args)
        return [], {"status": "FAILED", "proposals": []}

    monkeypatch.setattr(solver, "anneal", sample)
    monkeypatch.setattr(solver.llm, "request", request)
    result = solver.search_block(whole.full, [whole.best.point[whole.full.binary]], options,
                                 whole.deadline, 1, 1, whole)
    assert seeds == [options.seed]*3 and whole.result.llm_calls == expected
    assert len(requests) == expected and len(result.history) == 3
    assert [row["inner"] for row in whole.result.llm_history] == ([1]*3 + [2]*3 + [3]*3)[:expected]
    if expected > 1:
        assert requests[1][5]["previous_call_feedback"]["failed"]


def test_late_sa_candidates_not_dispatched(tiny, options, monkeypatch):
    d, s = tiny(1); whole = make_whole(d, s, options); b = whole.full
    now = [0.]; calls = []
    whole.deadline = 10
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    original_dispatch = solver.dispatch
    force_open_master(monkeypatch)

    def dispatch(*args):
        calls.append(1)
        return original_dispatch(*args)

    def late(*args):
        now[0] = 11.
        return [np.zeros(len(b.binary))], {"bits": 1}

    monkeypatch.setattr(solver, "dispatch", dispatch)
    monkeypatch.setattr(solver, "anneal", late)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
    assert len(calls) == 1 and result.upper == 225 and not result.history
    assert not whole.diagnostics["proposals"]


def test_sa_repair_crosses_boundary_and_freezes_batch_base(tiny, options, monkeypatch):
    d, s = tiny(2, 6)
    d.initial_on[:] = 1; d.initial_duration[:] = 10; d.initial_output[:] = 10
    d.min_up[:] = 4; d.min_down[:] = 3
    whole = make_whole(d, s, options)
    b = partition(whole.m, 3)[0][0]
    seed = whole.best.point[b.cols[b.binary]]
    a = seed.copy(); a[b.u[0, 1:]] = 0  # Deliberately leave inconsistent raw v/w.
    options.max_inner = 1
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, "anneal", lambda *args: ([a, seed.copy()], {"bits": 1}))
    result = solver.search_block(b, [seed], options, whole.deadline, 1, 1, whole)
    first, second = whole.diagnostics["proposals"]
    assert not first["raw_discrete_feasible"]
    assert first["repaired_u"][0][3] == 0  # Minimum downtime extends outside the block.
    assert second["raw_u"][0][3] == 1     # Still uses the batch-start incumbent.
    assert first["outcome"] == "accepted" and result.best is not None


def test_late_full_evaluation_does_not_update_incumbent(tiny, options, monkeypatch):
    d, s = tiny(2); whole = make_whole(d, s, options)
    old_cost = whole.result.objective; now = [0.]; whole.deadline = 10
    original = solver.dispatch
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        r = original(*args); now[0] = 11.
        return r

    monkeypatch.setattr(solver, "dispatch", late)
    raw = np.ones((2, 3)); raw[0] = 0
    assert not whole.accept(pack(d, raw), "sa", {})
    assert whole.result.objective == old_cost and len(whole.cache) == 1


def test_repricing_rebuilds_cuts_and_costs(tiny, options, monkeypatch):
    d, s = tiny(2, 5); whole = make_whole(d, s, options)
    b = partition(whole.m, 3)[0][0]
    options.max_inner = 1
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    first = solver.search_block(b, [whole.best.point[b.cols[b.binary]]], options, whole.deadline, 1, 1, whole)
    b2 = partition(whole.m, 3, np.full(len(whole.m.rhs), .25))[0][0]
    seen = []
    original = solver.master

    def inspect(block, cuts, seconds):
        seen.append(list(cuts))
        return original(block, cuts, seconds)

    monkeypatch.setattr(solver, "master", inspect)
    second = solver.search_block(b2, first.seeds, options, whole.deadline, 2, 1, whole)
    assert second.best.cost != first.best.cost
    assert len(seen[0]) == len(first.seeds)
    assert len(whole.diagnostics["local_evaluations"]) == 2


def test_unconverged_block_stops_before_coordination(tiny, options, monkeypatch):
    d, s = tiny(2, 5)
    options.max_inner = 1
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    monkeypatch.setattr(solver, "coordinate", lambda *args: (_ for _ in ()).throw(AssertionError("must not coordinate")))
    r = solver.solve(d, s, options)
    assert r.status == "INNER_NOT_CONVERGED" and not r.outer_history
    assert {entry["block"] for entry in r.history} == {1}


def test_closed_blocks_still_coordinate(tiny, options, monkeypatch):
    d, s = tiny(1, 5)
    d.initial_on[:] = 1; d.initial_duration[:] = 1; d.initial_output[:] = 10
    d.min_up[:] = 10
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    r = solver.solve(d, s, options)
    assert r.status == "DUAL_CONVERGED", r.message
    assert r.validation["feasible"] and r.outer_history


def test_master_solves_only_continuous_relaxation(tiny, monkeypatch):
    from suc_benders import benders
    d, s = tiny(); b = partition(model(d, s), 3)[0][0]
    calls = []
    original = benders.optimize

    def check(m, seconds, integral=False):
        calls.append(integral)
        return original(m, seconds, integral)

    monkeypatch.setattr(benders, "optimize", check)
    m, lower = benders.master(b, [], 10)
    assert calls == [False] and np.isfinite(lower) and not hasattr(m, "x")
