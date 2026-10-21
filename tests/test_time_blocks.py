import numpy as np
import pytest

from suc_benders import solver, search as search_core
from suc_benders.model import partition
from suc_benders.validation import commitment, pack
from conftest import make_whole


@pytest.fixture(autouse=True)
def unavailable_reference(monkeypatch):
    monkeypatch.setattr(solver, 'solve_reference', lambda *a:
                        dict(status='TIME_LIMIT', objective=None, validation=None, elapsed_seconds=0.))


def search_block(b, seeds, options, deadline, outer, index, whole):
    whole.deadline = deadline
    search = search_core.BlockSearch(b, options, outer, index, whole)
    for inner in range(1, options.max_inner + 1):
        search.step(seeds, inner)
        if whole.remaining() <= 0 or search_core.closed(search.lower, search.upper, options.tolerance):
            break
    return search


def force_open_master(monkeypatch):
    original = search_core.master
    monkeypatch.setattr(search_core, "master", lambda b, cuts, seconds=0, **kwargs: (original(b, cuts, **kwargs)[0], -1e6))


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
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([a, seed.copy()], {"bits": 1}))
    seen, prepare = [], whole.prepare
    def spy(raw, source, record):
        x = prepare(raw, source, record)
        seen.append((np.array(raw), commitment(d, x).u))
        return x
    monkeypatch.setattr(whole, "prepare", spy)
    result = search_block(b, [seed], options, whole.deadline, 1, 1, whole)
    first, second = whole.diagnostics["proposals"]
    assert not first["raw_discrete_feasible"]
    assert seen[0][1][0][3] == 0  # Minimum downtime extends outside the block.
    assert seen[1][0][0][3] == 1  # Still uses the batch-start incumbent.
    assert first["outcome"] == "accepted" and result.best is not None


def test_late_full_evaluation_does_not_update_incumbent(tiny, options, monkeypatch):
    d, s = tiny(2); whole = make_whole(d, s, options)
    old_cost = whole.result.objective; now = [0.]; whole.deadline = 10
    original = search_core.dispatch
    monkeypatch.setattr(search_core, "perf_counter", lambda: now[0])
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args, **kwargs):
        r = original(*args, **kwargs); now[0] = 11.
        return r

    monkeypatch.setattr(search_core, "dispatch", late)
    raw = np.ones((2, 3)); raw[0] = 0
    assert not whole.accept(pack(d, raw), "sa", {})
    assert whole.result.objective == old_cost and len(whole.cache) == 1


def test_unconverged_blocks_rotate_and_coordinate(tiny, options, monkeypatch):
    d, s = tiny(2, 24)
    options.max_inner = 2; options.max_outer = 2
    force_open_master(monkeypatch)
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([], {"bits": 1}))
    coordinate = solver.coordinate
    groups = []

    def inspect(planes, *args):
        groups.append([len(group) for group in planes])
        return coordinate(planes, *args)

    monkeypatch.setattr(solver, "coordinate", inspect)
    r = solver.solve(d, s, options)
    assert r.status == "SEARCH_EXHAUSTED", r.message
    assert [entry["block"] for entry in r.history] == list(range(1, 9))*4
    assert r.diagnostics["complete_rounds"] == 4 and r.diagnostics["coordination_count"] == 2
    assert len(groups) == 2 and all(len(group) == 8 and all(group) for group in groups)
    assert all(row["unconverged_blocks"] == list(range(1, 9)) for row in r.outer_history)
    assert all(row["visits"] == row["sa_calls"] == 2 for row in r.diagnostics["blocks"])
    assert r.validation["feasible"] and r.reference_cost is None
    assert all('lower' not in row for row in r.outer_history)


def test_round_rotation_retains_local_cuts_and_cache(tiny, options, monkeypatch):
    d, s = tiny(2, 6); whole = make_whole(d, s, options)
    searches = [search_core.BlockSearch(b, options, 1, j+1, whole)
                for j, b in enumerate(partition(whole.m, 3)[0])]
    force_open_master(monkeypatch)
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([], {"bits": 1}))
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
    monkeypatch.setattr(search_core, "perf_counter", lambda: now[0])
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    force_open_master(monkeypatch)

    def late(*args):
        now[0] = 11.
        return [], {"bits": 1}

    monkeypatch.setattr(search_core, "anneal", late)
    monkeypatch.setattr("suc_benders.upper_advisor.advisor_request.AdvisorRequest", lambda *args: pytest.fail("incomplete round request"))
    r = solver.solve(d, s, options)
    assert r.status == "TIME_LIMIT" and r.diagnostics["complete_rounds"] == 0
    assert not r.diagnostics["rounds"][0]["complete"] and not r.outer_history
    assert [row["visits"] for row in r.diagnostics["blocks"]] == [1, 0]
    assert r.llm_calls == 0 and r.validation["feasible"]


def test_coordination_failure_keeps_existing_failure_status(tiny, options, monkeypatch):
    options.max_outer = options.max_inner = 1
    force_open_master(monkeypatch)
    monkeypatch.setattr(search_core, 'anneal', lambda *a: ([], {'bits': 1}))
    def failed(*args):
        raise RuntimeError('Multiplier selection LP failed: INFEASIBLE')
    monkeypatch.setattr(solver, 'coordinate', failed)
    result = solver.solve(*tiny(2, 6), options)
    assert result.status == 'COORDINATION_FAILED'
    assert 'Multiplier selection LP failed' in result.message
    assert result.validation['feasible'] and result.reference_cost is None


def test_exhausted_sa_budget_skips_search_without_waiting(tiny, options):
    options.sa_budget_seconds = 0.
    result = solver.solve(*tiny(2, 6), options)
    assert result.status == 'SEARCH_EXHAUSTED' and result.diagnostics['sa_budget_exhausted']
    assert result.diagnostics['sa_calls'] == 0 and result.validation['feasible']
