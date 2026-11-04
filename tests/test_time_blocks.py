from types import SimpleNamespace as NS

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


def search_block(b, seeds, options, deadline, index, whole, rounds=1):
    whole.deadline = deadline
    search = search_core.BlockSearch(b, options, index, whole)
    for k in range(1, rounds + 1):
        search.step(seeds, k)
        if whole.remaining() <= 0:
            break
    return search


def test_sa_repair_crosses_boundary_and_freezes_batch_base(tiny, options, monkeypatch):
    d, s = tiny(2, 6)
    d.initial_on[:] = 1; d.initial_duration[:] = 10; d.initial_output[:] = 10
    d.min_up[:] = 4; d.min_down[:] = 3
    whole = make_whole(d, s, options)
    b = partition(whole.m, 3)[0][0]
    seed = whole.best.point[b.cols[b.binary]]
    a = seed.copy(); a[b.u[0, 1:]] = 0  # Deliberately leave inconsistent raw v/w.
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([a, seed.copy()], {"bits": 1}))
    seen, prepare = [], whole.prepare
    def spy(raw, source, record):
        x = prepare(raw, source, record)
        seen.append((np.array(raw), commitment(d, x).u))
        return x
    monkeypatch.setattr(whole, "prepare", spy)
    result = search_block(b, [seed], options, whole.deadline, 1, whole)
    first, second = whole.diagnostics["proposals"]
    assert not first["raw_discrete_feasible"]
    assert seen[0][1][0][3] == 0  # Minimum downtime extends outside the block.
    assert seen[1][0][0][3] == 1  # Still uses the batch-start incumbent.
    assert first["outcome"] == "accepted" and result.best_local is not None


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


def test_persistent_blocks_rotate_every_round(tiny, options, monkeypatch):
    d, s = tiny(2, 24)
    options.max_rounds = 4
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([], {"bits": 1}))
    built = []
    original = solver.BlockSearch
    monkeypatch.setattr(solver, "BlockSearch", lambda *args: built.append(args[2]) or original(*args))
    r = solver.solve(d, s, options)
    assert r.status == "SEARCH_EXHAUSTED", r.message
    assert r.diagnostics["search_stop_reason"] == "MAX_ROUNDS"
    # 每块只建立一次搜索对象，按轮依次访问；不再有外层重建或闭合跳过。
    assert built == list(range(1, 9))
    assert [entry["block"] for entry in r.history] == list(range(1, 9))*4
    assert [entry["round"] for entry in r.history] == [k for k in range(1, 5) for _ in range(8)]
    assert r.diagnostics["complete_rounds"] == 4 and len(r.diagnostics["blocks"]) == 8
    assert all(row["visits"] == row["sa_calls"] == 4 for row in r.diagnostics["blocks"])
    assert r.validation["feasible"] and r.reference_cost is None


def test_round_rotation_retains_local_cuts_and_cache(tiny, options, monkeypatch):
    d, s = tiny(2, 6); whole = make_whole(d, s, options)
    searches = [search_core.BlockSearch(b, options, j+1, whole)
                for j, b in enumerate(partition(whole.m, 3)[0])]
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([], {"bits": 1}))
    for search in searches:
        search.step([whole.best.point[search.b.cols[search.b.binary]]], 1)
    first = [(list(x.cuts), dict(x.cache), x.best_local) for x in searches]
    evaluations = len(whole.diagnostics["local_evaluations"])
    for search, (cuts, cache, best) in zip(searches, first):
        search.step([whole.best.point[search.b.cols[search.b.binary]]], 2)
        assert len(search.cuts) == len(cuts) and all(a is b for a, b in zip(search.cuts, cuts))
        assert all(search.cache[k] is v for k, v in cache.items()) and search.best_local is best
    assert len(whole.diagnostics["local_evaluations"]) == evaluations


def test_incomplete_round_is_not_counted(tiny, options, monkeypatch):
    d, s = tiny(2, 6); now = [0.]
    options.time_limit = 10
    monkeypatch.setattr(search_core, "perf_counter", lambda: now[0])
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 11.
        return [], {"bits": 1}

    monkeypatch.setattr(search_core, "anneal", late)
    r = solver.solve(d, s, options)
    assert r.status == "TIME_LIMIT" and r.diagnostics["complete_rounds"] == 0
    assert not r.diagnostics["rounds"][0]["complete"]
    assert [row["visits"] for row in r.diagnostics["blocks"]] == [1, 0]
    assert r.validation["feasible"]


def test_exhausted_sa_budget_skips_search_without_waiting(tiny, options):
    options.sa_budget_seconds = 0.
    result = solver.solve(*tiny(2, 6), options)
    assert result.status == 'SEARCH_EXHAUSTED' and result.diagnostics['sa_budget_exhausted']
    assert result.diagnostics['sa_calls'] == 0 and result.validation['feasible']


def test_failure_after_deadline_keeps_stage_name(tiny, options, monkeypatch):
    now = [0.]; options.time_limit = 10
    monkeypatch.setattr(search_core, "perf_counter", lambda: now[0])
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def broken(*args):
        now[0] = 11.
        raise RuntimeError("boom")

    monkeypatch.setattr(search_core, "anneal", broken)
    result = solver.solve(*tiny(2, 6), options)
    # 超过截止时间后抛出的非 TimeoutError 是真实故障，不能被记成 TIME_LIMIT。
    assert result.status == "ROUND_1_BLOCK_1_FAILED"
    assert result.message == "RuntimeError: boom" and result.validation["feasible"]


def test_qubo_encodes_only_the_feasibility_cuts_closest_to_the_anchor(tiny, options, monkeypatch):
    whole = make_whole(*tiny(2, 6), options)
    search = search_core.BlockSearch(partition(whole.m, 3)[0][1], options, 2, whole)
    monkeypatch.setattr(search_core, "QUBO_FEASIBILITY_CUTS", 3)
    nx = len(search.b.binary)
    anchor = np.ones(nx)
    optimality = [NS(optimality=True, alpha=-float(k), beta=np.zeros(nx)) for k in range(2)]
    feasibility = [NS(optimality=False, alpha=-10.+k, beta=np.zeros(nx)) for k in range(6)]
    for cut in [optimality[0], *feasibility[:3], optimality[1], *feasibility[3:]]:
        search.add_cut(cut)
    cuts, stored, encoded = search.qubo_cuts(anchor)
    # 保留全部最优性割和 alpha+beta·anchor 最大（松弛量最小）的 3 条可行割，顺序不变。
    assert (stored, encoded) == (6, 3)
    assert cuts == [optimality[0], optimality[1], *feasibility[3:]]


def test_cut_store_is_first_in_first_out_and_totals_keep_counting(tiny, options, monkeypatch):
    whole = make_whole(*tiny(2, 6), options)
    search = search_core.BlockSearch(partition(whole.m, 3)[0][0], options, 1, whole)
    monkeypatch.setattr(search_core, "CUT_LIMIT", 3)
    cuts = [NS(optimality=k % 2 == 0, alpha=0., beta=np.zeros(1)) for k in range(5)]
    for cut in cuts:
        search.add_cut(cut)
    search.summarize()
    assert search.cuts == cuts[2:] and search.diagnostic["cuts_dropped"] == 2
    assert search.cut_totals == dict(optimality=3, feasibility=2)
    assert search.diagnostic["stored_cuts"] == 3 and search.diagnostic["optimality_cuts"] == 3


def test_block_cache_keeps_compact_records_and_best_local_keeps_the_point(tiny, options, monkeypatch):
    whole = make_whole(*tiny(2, 6), options)
    search = search_core.BlockSearch(partition(whole.m, 3)[0][0], options, 1, whole)
    monkeypatch.setattr(search_core, "anneal", lambda *args: ([], {"bits": 1}))
    search.step([whole.best.point[search.b.cols[search.b.binary]]], 1)
    record, = search.cache.values()
    assert set(vars(record)) == {"feasible", "status", "cost"}
    assert search.best_local.point is not None and search.best_local.cost == record.cost
