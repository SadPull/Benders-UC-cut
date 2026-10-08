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


def search_block(b, seeds, options, deadline, outer, index, whole):
    whole.deadline = deadline
    search = solver.BlockSearch(b, options, outer, index, whole)
    for inner in range(1, options.max_inner + 1):
        search.step(seeds, inner)
        if whole.remaining() <= 0 or solver.closed(search.lower, search.upper, options.tolerance):
            break
    return search


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
    result = search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
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
    result = search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
    assert result.lower == initial_bound and result.best is not None


@pytest.mark.parametrize("max_calls,expected", [(2, 2), (0, 0), (None, 3)])
def test_llm_failure_returns_to_full_sa_round_and_fixed_seed(tiny, options, monkeypatch, max_calls, expected):
    d, s = tiny(2, 5)
    options.llm_enabled = True; options.llm_patience = 1; options.llm_max_calls = max_calls
    options.max_inner = 3; options.max_outer = 1
    force_open_master(monkeypatch)
    seeds, requests, events = [], [], []

    def sample(q, o, seconds, seed):
        seeds.append(seed); events.append("sa")
        return [], {"bits": 1}

    def request(*args):
        requests.append(args); events.append("llm")
        return [], {"status": "FAILED", "proposals": []}

    monkeypatch.setattr(solver, "anneal", sample)
    monkeypatch.setattr(solver.llm, "request", request)
    result = solver.solve(d, s, options)
    assert result.status == "MAX_ITERATIONS", result.message
    assert seeds == [options.seed]*6 and result.llm_calls == expected
    assert events == [event for i in range(3) for event in
                      (["sa", "sa", "llm"] if i < expected else ["sa", "sa"])]
    assert len(requests) == expected and len(result.history) == 6
    assert [row["round"] for row in result.llm_history] == list(range(1, expected+1))
    assert all(row["block"] is None and row["stage"] == "round_end" for row in result.llm_history)
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
    result = search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 1, 1, whole)
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
    result = search_block(b, [seed], options, whole.deadline, 1, 1, whole)
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
    first = search_block(b, [whole.best.point[b.cols[b.binary]]], options, whole.deadline, 1, 1, whole)
    b2 = partition(whole.m, 3, np.full(len(whole.m.rhs), .25))[0][0]
    seen = []
    original = solver.master

    def inspect(block, cuts, seconds):
        seen.append(list(cuts))
        return original(block, cuts, seconds)

    monkeypatch.setattr(solver, "master", inspect)
    second = search_block(b2, first.seeds, options, whole.deadline, 2, 1, whole)
    assert second.best.cost != first.best.cost
    assert len(seen[0]) == len(first.seeds)
    assert len(whole.diagnostics["local_evaluations"]) == 2


def test_unconverged_blocks_rotate_and_coordinate(tiny, options, monkeypatch):
    d, s = tiny(2, 24)
    options.max_inner = 2; options.max_outer = 2
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    coordinate = solver.coordinate
    groups = []

    def inspect(planes, *args):
        groups.append([len(group) for group in planes])
        return coordinate(planes, *args)

    monkeypatch.setattr(solver, "coordinate", inspect)
    r = solver.solve(d, s, options)
    assert r.status == "MAX_ITERATIONS", r.message
    assert [entry["block"] for entry in r.history] == list(range(1, 9))*4
    assert r.diagnostics["complete_rounds"] == 4 and r.diagnostics["coordination_count"] == 2
    assert len(groups) == 2 and all(len(group) == 8 and all(group) for group in groups)
    assert all(row["unconverged_blocks"] == list(range(1, 9)) for row in r.outer_history)
    assert all(row["visits"] == row["sa_calls"] == 2 for row in r.diagnostics["blocks"])
    assert r.validation["feasible"] and r.lower_bound <= r.objective + 1e-5
    assert all(a["lower"] <= b["lower"] for a, b in zip(r.outer_history, r.outer_history[1:]))


def test_round_rotation_retains_local_cuts_and_cache(tiny, options, monkeypatch):
    d, s = tiny(2, 6); whole = make_whole(d, s, options)
    searches = [solver.BlockSearch(b, options, 1, j+1, whole)
                for j, b in enumerate(partition(whole.m, 3)[0])]
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    for search in searches:
        search.step([whole.best.point[search.b.cols[search.b.binary]]], 1)
    first = [(list(x.cuts), dict(x.cache), x.best) for x in searches]
    evaluations = len(whole.diagnostics["local_evaluations"])
    for search, (cuts, cache, best) in zip(searches, first):
        search.step([whole.best.point[search.b.cols[search.b.binary]]], 2)
        assert len(search.cuts) == len(cuts) and all(a is b for a, b in zip(search.cuts, cuts))
        assert all(search.cache[k] is v for k, v in cache.items()) and search.best is best
    assert len(whole.diagnostics["local_evaluations"]) == evaluations


def test_incomplete_round_neither_counts_nor_requests_advice(tiny, options, monkeypatch):
    d, s = tiny(2, 6); now = [0.]
    options.time_limit = 10; options.llm_enabled = True; options.llm_patience = 1
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    force_open_master(monkeypatch)

    def late(*args):
        now[0] = 11.
        return [], {"bits": 1}

    monkeypatch.setattr(solver, "anneal", late)
    monkeypatch.setattr(solver.llm, "request", lambda *args: pytest.fail("incomplete round request"))
    r = solver.solve(d, s, options)
    assert r.status == "TIME_LIMIT" and r.diagnostics["complete_rounds"] == 0
    assert not r.diagnostics["rounds"][0]["complete"] and not r.outer_history
    assert [row["visits"] for row in r.diagnostics["blocks"]] == [1, 0]
    assert r.llm_calls == 0 and r.validation["feasible"]


def test_patience_counts_complete_rounds(tiny, options, monkeypatch):
    d, s = tiny(2, 6)
    options.max_outer = 1; options.max_inner = 7; options.llm_patience = 3; options.llm_enabled = True
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    monkeypatch.setattr(solver.llm, "request", lambda *args: ([], {"status": "FAILED", "proposals": []}))
    r = solver.solve(d, s, options)
    assert [row["round"] for row in r.llm_history] == [3, 6]
    assert r.diagnostics["complete_rounds"] == 7 and r.diagnostics["sa_calls"] == 14


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
