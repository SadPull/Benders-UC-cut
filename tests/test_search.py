from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import solver
from suc_benders.model import model, partition
from suc_benders.validation import pack


@pytest.fixture(autouse=True)
def proof_after_upper_search(monkeypatch):
    backend = solver.StrengthenedCertification
    monkeypatch.setattr(backend, 'start', lambda self: None)
    monkeypatch.setattr(backend, 'close', lambda self: None)
    def step(cert):
        if cert.whole.diagnostics['coordination_count'] >= cert.o.max_outer:
            cert.update_bound(cert.result.objective)
    monkeypatch.setattr(backend, 'step', step)


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

    def late(*args, **kwargs):
        r = original(*args, **kwargs); now[0] = 11.
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
    assert r.status == "CONVERGED", r.message
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
    monkeypatch.setattr("suc_benders.upper_advisor.advisor_request.AdvisorRequest", lambda *args: pytest.fail("incomplete round request"))
    r = solver.solve(d, s, options)
    assert r.status == "TIME_LIMIT" and r.diagnostics["complete_rounds"] == 0
    assert not r.diagnostics["rounds"][0]["complete"] and not r.outer_history
    assert [row["visits"] for row in r.diagnostics["blocks"]] == [1, 0]
    assert r.llm_calls == 0 and r.validation["feasible"]


def test_initial_global_convergence_skips_search(tiny, options, monkeypatch):
    d, s = tiny(1, 5)
    d.initial_on[:] = 1; d.initial_duration[:] = 1; d.initial_output[:] = 10
    d.min_up[:] = 10
    monkeypatch.setattr(solver, "anneal", lambda *args: ([], {"bits": 1}))
    r = solver.solve(d, s, options)
    assert r.status == "CONVERGED", r.message
    assert r.validation["feasible"] and not r.outer_history
    assert r.diagnostics['sa_calls'] == 0


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


def uninitialized_whole(d, scenarios, options):
    m = model(d, scenarios)
    started = perf_counter()
    return solver.WholeDay(m, partition(m, d.horizon)[0][0], options,
                           started+60, solver.Result(lower_bound=0.), started)


def test_missing_candidate_preserves_rejection_reason(tiny, options):
    whole = uninitialized_whole(*tiny(1), options)
    record = {}
    x = whole.prepare(np.ones((1, 2)), "initial", record)
    assert record["outcome"] == "invalid_matrix"
    assert not whole.accept(x, "initial", record)
    assert record["outcome"] == "invalid_matrix" and whole.remaining() > 0


def test_repaired_discrete_flag_is_checked(tiny, options, monkeypatch):
    d, scenarios = tiny(1)
    d.min_down[:] = 3
    whole = uninitialized_whole(d, scenarios, options)
    monkeypatch.setattr(solver, "repair", lambda *args: pack(d, np.ones((1, 3))))
    record = {}
    assert whole.prepare(np.ones((1, 3)), "sa", record) is None
    assert not record["repaired_discrete_feasible"]
    assert record["outcome"] == "repair_discrete_infeasible"


def test_first_upper_is_not_infinite_cost_saving(tiny, options):
    whole = uninitialized_whole(*tiny(1), options)
    record = {}
    assert whole.accept(pack(whole.d, np.ones((1, 3))), "initial", record)
    assert record["incumbent_decrease"] == 0.
    assert whole.diagnostics["initial_upper"] == whole.result.objective


def test_initialization_projection_recovers_feasible_case(tiny, options, monkeypatch):
    from suc_benders.model import fingerprint
    from suc_benders.validation import validate

    d, scenarios = tiny(3, 4)
    d.pmin[:] = [20., 20., 0.]
    d.pmax[:] = d.pstart[:] = d.pshut[:] = [100., 100., 5.]
    d.ramp_up[:] = d.ramp_down[:] = 100.
    d.cost_b[:] = [1., 1., 100.]
    d.cost_c[:] = d.startup_cost[:] = 1.
    d.shutdown_cost[:] = 0.
    d.base_load[:] = scenarios[0].load[:] = [5., 80., 5., 5.]
    whole = uninitialized_whole(d, scenarios, options)
    identity = fingerprint(whole.m)
    relaxation = solver.optimize(whole.m, 5)
    whole.result.lower_bound = relaxation.lower
    seen = []
    original = solver.optimize

    def project(m, seconds, **kwargs):
        seen.append(kwargs)
        assert not m.q.any() and seconds <= 30
        np.testing.assert_array_equal(m.obj[m.ix.u], 1-2*np.floor(relaxation.x[m.ix.u]+.5))
        raw = original(m, seconds, **kwargs)
        raw.lower = 1e20  # The distance bound must never become an original-problem LB.
        return raw

    monkeypatch.setattr(solver, "optimize", project)
    monkeypatch.setattr(solver, "anneal", lambda *a: pytest.fail("SA started"))
    monkeypatch.setattr(whole.certifier, "start", lambda: pytest.fail("Proof started"))
    monkeypatch.setattr(whole.advice, "start_request", lambda: pytest.fail("LLM started"))
    whole.initialize(relaxation)
    assert len(seen) == 1 and seen[0]["integral"] and seen[0]["solution_limit"] == 1
    assert whole.result.validation["feasible"] and validate(d, scenarios, whole.result)["feasible"]
    assert whole.result.lower_bound == relaxation.lower
    assert fingerprint(whole.m) == identity
    assert whole.diagnostics["evaluations"][-1]["source"] == "initialization_mip"
    assert whole.diagnostics["initialization_repair"]["outcome"] == "accepted"
    assert whole.diagnostics["sa_calls"] == whole.result.llm_calls == 0


@pytest.mark.parametrize("status,point,accepted", [
    ("SOLUTION_LIMIT", True, True), ("TIME_LIMIT", True, True),
    ("TIME_LIMIT", False, False), ("INFEASIBLE", False, False),
    ("NUMERIC", True, False), ("INTERRUPTED", True, False),
])
def test_initialization_projection_statuses(tiny, options, monkeypatch, status, point, accepted):
    whole = uninitialized_whole(*tiny(1), options)
    feasible = solver.dispatch(whole.full, pack(whole.d, np.ones((1, 3))), 5)
    relaxation = NS(x=feasible.point)
    prepare = whole.prepare
    monkeypatch.setattr(whole, "prepare", lambda u, source, record:
                        None if source == "initial" else prepare(u, source, record))
    monkeypatch.setattr(solver, "optimize", lambda *a, **k:
                        NS(status=status, x=feasible.point if point else None, lower=1e20))
    if accepted:
        whole.initialize(relaxation)
        assert whole.result.validation["feasible"]
    else:
        with pytest.raises(RuntimeError, match=status):
            whole.initialize(relaxation)
        assert not np.isfinite(whole.result.objective)
    assert whole.diagnostics["initialization_repair"]["status"] == status
    assert whole.result.lower_bound == 0.


@pytest.mark.parametrize("remaining", [4., 20., 60.])
@pytest.mark.parametrize("overrun", ["projection", "total"])
def test_initialization_projection_budget_and_late_result(tiny, options, monkeypatch, remaining, overrun):
    whole = uninitialized_whole(*tiny(1), options)
    point = solver.dispatch(whole.full, pack(whole.d, np.ones((1, 3))), 5).point
    ticks = [0.]
    whole.deadline = remaining
    monkeypatch.setattr(solver, "perf_counter", lambda: ticks[0])
    monkeypatch.setattr(whole, "prepare", lambda *a: None)
    calls = []

    def late(m, seconds, **kwargs):
        calls.append(seconds)
        assert seconds <= min(30., remaining-5.)
        ticks[0] = remaining+1 if overrun == "total" else min(30., remaining-5.)+1
        return NS(status="SOLUTION_LIMIT", x=point)

    monkeypatch.setattr(solver, "optimize", late)
    with pytest.raises(TimeoutError if remaining > 5 and overrun == "total" else RuntimeError):
        whole.initialize(NS(x=point))
    assert len(calls) == (0 if remaining <= 5 else 1)
    assert not np.isfinite(whole.result.objective) and not whole.cache


@pytest.mark.parametrize("source", ["sa", "llm"])
def test_shared_repair_never_starts_integer_projection(tiny, options, monkeypatch, source):
    d, scenarios = tiny(2)
    scenarios[0].load[:] = 10.
    whole = uninitialized_whole(d, scenarios, options)
    monkeypatch.setattr(solver, "optimize", lambda *a, **k: pytest.fail("Projection outside initialization"))
    record = {}
    x = whole.prepare(np.ones((2, 3)), source, record)
    assert x is None and not whole.accept(x, source, record)
    assert record["outcome"] == "repair_failed"
    assert "initialization_repair" not in whole.diagnostics


def test_initialization_skips_projection_with_valid_upper(tiny, options, monkeypatch):
    whole = uninitialized_whole(*tiny(1), options)
    relaxation = solver.optimize(whole.m, 5)
    monkeypatch.setattr(solver, "optimize", lambda *a, **k: pytest.fail("Unneeded MIP projection"))
    whole.initialize(relaxation)
    assert whole.result.validation["feasible"]
    assert "initialization_repair" not in whole.diagnostics


def exchange_whole(tiny, options):
    d, scenarios = tiny(2, 6)
    d.pmax[:] = d.pstart[:] = d.pshut[:] = 100.
    d.pmin[:] = 40.; d.min_up[:] = 4.; d.cost_b[:] = [10., 1.]
    scenarios[0].load[:] = 60.
    options.enhanced_search = True
    whole = uninitialized_whole(d, scenarios, options)
    u = np.zeros((2, 6)); u[0] = 1.
    assert whole.accept(pack(d, u), 'initial')
    return whole


def test_joint_exchange_crosses_blocks_without_mip_or_bound_feedback(tiny, options, monkeypatch):
    from suc_benders.model import fingerprint
    whole = exchange_whole(tiny, options)
    before, lower, identity = whole.result.objective, whole.result.lower_bound, fingerprint(whole.m)
    u = 1-whole.result.u
    np.testing.assert_array_equal(whole.prepare(np.zeros_like(u), 'sa', {}), pack(whole.d, whole.result.u))
    assert whole.prepare(np.ones_like(u), 'sa', {}) is None
    seen = []
    optimize = solver.optimize
    def lp(m, seconds, **kwargs):
        assert not kwargs.get('integral', False)
        r = optimize(m, seconds, **kwargs); r.lower = 1e20
        return r
    def sample(q, o, seconds, seed):
        seen.append(seed)
        return [pack(whole.d, u), pack(whole.d, 1-u)], {'bits': q.Q.shape[0]}
    monkeypatch.setattr(solver, 'optimize', lp)
    monkeypatch.setattr(solver, 'anneal', sample)
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    whole.joint_step([], 1, 3)
    assert whole.result.objective < before and whole.result.validation['feasible']
    assert whole.result.lower_bound == lower and fingerprint(whole.m) == identity
    assert whole.diagnostics['evaluations'][-1]['source'] == 'joint_sa'
    assert whole.diagnostics['joint_search']['stalled_rounds'] == 0
    assert whole.diagnostics['joint_sa_calls'] == whole.diagnostics['sa_calls'] == len(seen) == 1
    records = whole.diagnostics['proposals'][-2:]
    np.testing.assert_array_equal(records[0]['raw_u'], u)
    np.testing.assert_array_equal(records[1]['raw_u'], 1-u)
    assert whole.diagnostics['joint_batches'][0]['hours'] == list(range(1, 7))


def test_joint_uses_incumbent_instead_of_worse_anchor(tiny, options, monkeypatch):
    d, scenarios = tiny(3, 6)
    d.pmax[:] = d.pstart[:] = d.pshut[:] = [60., 20., 20.]
    d.cost_a[:] = 0.; d.cost_b[:] = [10., 1., 1.]; d.cost_c[:] = [0., 200., 0.]
    options.enhanced_search = True
    whole = uninitialized_whole(d, scenarios, options)
    a = np.zeros((3, 6)); a[0] = 1.
    b = a.copy(); b[1] = 1.
    c = b.copy(); c[0] = 0.; c[2] = 1.
    assert whole.accept(pack(d, a), 'initial')
    before = whole.result.objective
    assert whole.accept(pack(d, b), 'sa') and whole.cache[pack(d, b).astype(np.uint8).tobytes()]['objective'] > before
    assert whole.result.objective == before
    whole.diagnostics['sa_calls'] = 7
    base, base_id, cost = whole.search_anchor()
    np.testing.assert_array_equal(base, b)
    assert cost > before
    monkeypatch.setattr(solver, 'select_units', lambda *args: [0, 1, 2])
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([pack(d, c)], {'bits': 1}))
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    whole.joint_step([], 1, 3)
    assert whole.result.objective < before
    assert whole.diagnostics['joint_batches'][0]['anchor'] != base_id
    np.testing.assert_array_equal(whole.diagnostics['joint_batches'][0]['baseline_u'], a)
    np.testing.assert_array_equal(whole.result.u, c)


def test_joint_rejects_repair_outside_selected_units(tiny, options, monkeypatch):
    d, scenarios = tiny(3, 4); options.enhanced_search = True
    whole = make_whole(d, scenarios, options)
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    monkeypatch.setattr(solver, 'select_units', lambda *args: [0, 1])
    raw = np.ones((3, 4)); raw[0, -1] = 0.
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([pack(d, raw)], {'bits': 1}))
    repaired = raw.copy(); repaired[0] = 0.; repaired[2] = 0.
    monkeypatch.setattr(solver, 'repair', lambda *args: pack(d, repaired))
    before, count = whole.result.objective, len(whole.cache)
    whole.joint_step([], 1, 3)
    assert whole.diagnostics['proposals'][-1]['outcome'] == 'repair_left_neighborhood'
    assert whole.result.objective == before and len(whole.cache) == count


def test_repair_memo_preserves_failure_origin_and_checks_deadline(tiny, options, monkeypatch):
    options.enhanced_search = True
    whole = uninitialized_whole(*tiny(), options)
    calls = []
    monkeypatch.setattr(solver, 'repair', lambda *args: calls.append(1))
    for source in ('sa', 'joint_sa'):
        record = {'proposal_id': source}
        assert whole.prepare(np.ones((2, 3)), source, record) is None
        assert record['origin'] == source and record['proposal_id'] == source
        assert record['outcome'] == 'repair_failed'
    assert len(calls) == 1 and record['repair_cache_hit']
    whole.deadline = 0.
    record = {}
    assert whole.prepare(np.ones((2, 3)), 'sa', record) is None
    assert record['outcome'] == 'deadline' and len(calls) == 1


def test_repair_memo_copies_and_reprices_each_block(tiny, options, monkeypatch):
    options.enhanced_search = True
    whole = make_whole(*tiny(), options)
    raw = np.ones((2, 3)); raw[0] = 0.
    x = whole.prepare(raw, 'sa', {})
    saved = x.copy(); x[:] = 0.
    record = {}; x = whole.prepare(raw, 'joint_sa', record)
    np.testing.assert_array_equal(x, saved)
    first = solver.BlockSearch(whole.full, options, 1, 1, whole)
    r1 = first.evaluate(x, 'sa')
    b2 = partition(whole.m, 3, np.full(len(whole.m.rhs), .25))[0][0]
    second = solver.BlockSearch(b2, options, 2, 1, whole)
    second.evaluate(x, 'sa')
    assert record['repair_cache_hit'] and r1 is not None
    assert first.context != second.context and len(whole.diagnostics['local_evaluations']) == 2


def test_joint_pairs_and_triples_start_without_stagnation(tiny, options, monkeypatch):
    d, scenarios = tiny(6, 4); d.pmin[:] = 0.; options.enhanced_search = True
    whole = make_whole(d, scenarios, options)
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([], {'bits': 1}))
    whole.joint_step([], 1, 1)
    assert len(whole.diagnostics['joint_batches']) == 1
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    for inner in range(3, 7):
        whole.joint_step([], 1, inner)
    assert [len(b['units']) for b in whole.diagnostics['joint_batches']] == [2, 3, 2, 3, 2]
    assert len({b['seed'] for b in whole.diagnostics['joint_batches']}) == 5
    assert whole.diagnostics['joint_search']['attempts'] == 5


def test_two_joint_batches_per_block_precede_round_advice(tiny, options, monkeypatch):
    options.enhanced_search = True; options.max_inner = 4; options.max_outer = 1
    force_open_master(monkeypatch)
    monkeypatch.setattr(solver, 'anneal', lambda *a: ([], {'bits': 1}))
    events = []
    def joint(whole, searches, outer, inner):
        events.append(('joint', inner))
    monkeypatch.setattr(solver.WholeDay, 'joint_step', joint)
    monkeypatch.setattr(solver.UpperAdvisor, 'after_round',
                        lambda self, before, sa_before, searches, outer, inner, allow: events.append(('advice', inner)))
    result = solver.solve(*tiny(2, 6), options)
    assert result.status == 'CONVERGED', result.message
    assert events == [event for i in range(1, 5) for event in [*[('joint', i)]*4, ('advice', i)]]


@pytest.mark.parametrize('case', ['budget', 'memory', 'late_lp', 'late_sa', 'INFEASIBLE', 'NUMERIC', 'bound'])
def test_joint_limits_keep_upper_and_lower(tiny, options, monkeypatch, case):
    whole = exchange_whole(tiny, options)
    before, lower = whole.result.objective, whole.result.lower_bound
    ticks = [0.]; whole.deadline = 30.
    monkeypatch.setattr(solver, 'perf_counter', lambda: ticks[0])
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    if case == 'budget':
        ticks[0] = 26.
    if case == 'memory':
        monkeypatch.setattr(solver, 'qubo', lambda *args: NS(Q=NS(shape=(4000, 4000))))
    optimize = solver.optimize
    def lp(m, seconds):
        if case in ('INFEASIBLE', 'NUMERIC'):
            return NS(status=case)
        r = optimize(m, seconds)
        if case == 'late_lp': ticks[0] = 31.
        if case == 'bound': r.objval = before+1.
        return r
    monkeypatch.setattr(solver, 'optimize', lp)
    def sample(*args):
        assert case == 'late_sa'
        ticks[0] = 31.
        return [pack(whole.d, 1-whole.result.u)], {'bits': 1}
    monkeypatch.setattr(solver, 'anneal', sample)
    whole.joint_step([], 1, 3)
    assert whole.result.objective == before and whole.result.lower_bound == lower
    assert len(whole.cache) == 1 and not whole.diagnostics['proposals']
    assert whole.diagnostics['joint_batches'][-1]['status'] != 'COMPLETE'
    assert whole.diagnostics['sa_calls'] == (1 if case == 'late_sa' else 0)


def test_anchor_selection_is_cost_diverse_and_sampling_reproducible(tiny, options):
    import hashlib
    options.enhanced_search = True
    d, scenarios = tiny(3, 6)
    whole = uninitialized_whole(d, scenarios, options)
    for code in range(24):
        u = np.ones((3, 6)); u.ravel()[:5] = [(code >> bit) & 1 for bit in range(5)]
        key = pack(d, u).astype(np.uint8).tobytes()
        whole.cache[key] = dict(feasible=True, objective=float(code), u=u.tolist(),
                               commitment_hash=hashlib.sha256(key).hexdigest())
    anchors = whole.anchors()
    assert len(anchors) == len({key for key, _ in anchors}) == 16
    assert [r['objective'] for _, r in anchors[:4]] == [0., 1., 2., 3.]
    count = d.n_units*d.horizon
    selected = anchors[:4]
    candidates = sorted(whole.cache.items(), key=lambda item: (item[1]['objective'], item[1]['commitment_hash']))
    for chosen in anchors[4:]:
        remaining = [item for item in candidates if item[0] not in {key for key, _ in selected}]
        distances = [min(np.count_nonzero(np.frombuffer(key, np.uint8)[:count] != np.frombuffer(other, np.uint8)[:count])
                         for other, _ in selected) for key, _ in remaining]
        assert chosen[0] == remaining[int(np.argmax(distances))][0]
        selected.append(chosen)
    seeds = [whole.sampling_seed('block'), whole.sampling_seed('joint')]
    assert seeds[0] != seeds[1] and whole.sampling_seed('block') == seeds[0]
    whole.diagnostics['sa_calls'] += 1
    assert whole.sampling_seed('block') not in seeds
    options.enhanced_search = False
    assert whole.sampling_seed('block') == whole.sampling_seed('joint') == options.seed


def test_joint_master_physical_rows_and_fixed_complement(tiny):
    from fractions import Fraction
    from suc_benders.neighborhood import joint_master, fix_units
    from suc_benders.qubo import fixed_binary
    d, scenarios = tiny(3, 6)
    d.pmin[:] = [10., 10., .15]; d.ramp_up[0] = 9.
    d.initial_on[1] = 1.; d.initial_output[1] = 30.; d.ramp_down[1] = 20.
    d.min_up[1] = 5.; d.initial_duration[1] = 3.
    scenarios = [NS(name='low', probability=.5, load=np.full(6, 20.25)),
                 NS(name='high', probability=.5, load=np.full(6, 30.))]
    d.reserve_fraction = .1
    full = partition(model(d, scenarios), d.horizon)[0][0]
    guide = joint_master(full, [])
    fixed = fixed_binary(guide)
    assert (fixed[full.v[0]] == 0).all()
    assert fixed[full.w[1, 0]] == 0 and (fixed[full.u[1, :2]] == 1).all()
    assert (fixed[full.w[:, -1]] == 0).all()
    capacity = list(guide.capacity_rows.values())
    assert all(bound == Fraction('33') for _, bound in capacity[:6])
    assert all(bound == Fraction('-20.25') for _, bound in capacity[-6:])
    assert all(row[int(full.u[2, t])] == Fraction('.15') for t, (row, _) in enumerate(capacity[-6:]))
    u = np.ones((3, 6)); u[0] = 0.
    fix_units(guide, full, u, [2])
    fixed = fixed_binary(guide)
    assert (fixed[full.u[:2]] == u[:2]).all()
    assert np.isnan(fixed[full.u[2]]).any()
    assert solver.qubo(guide, NS(theta_step=1., penalty=1e6, reserve_scale=100.), 1e4).Q.shape[0] > 0


def test_joint_unit_coverage_and_no_movable_units(tiny, options, monkeypatch):
    from suc_benders.neighborhood import select_units
    d, scenarios = tiny(5, 4)
    groups = [select_units(np.arange(5), i, 17) for i in range(20)]
    assert len({tuple(group) for group in groups}) == 20
    assert groups == [select_units(np.arange(5), i, 17) for i in range(20)]
    whole = exchange_whole(tiny, options)
    whole.fixed[:] = 0.
    monkeypatch.setattr(solver, 'anneal', lambda *a: pytest.fail('No movable units'))
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    whole.joint_step([], 1, 3)
    assert whole.diagnostics['joint_batches'][-1]['status'] == 'NO_MOVABLE_UNITS'


@pytest.mark.parametrize('folder,scenario', [('data/classic10', 'deterministic.json'),
    ('data/classic10', 'stochastic.json'), ('tests/fixtures/rts24', 'deterministic.json'),
    ('data/rts24/inputs', 'stochastic.json')])
def test_joint_builds_on_classic_and_rts_without_changing_physics(options, monkeypatch, folder, scenario):
    from pathlib import Path
    from suc_benders.data import load_data
    from suc_benders.model import fingerprint
    root = Path(__file__).resolve().parents[1]
    d, scenarios = load_data(root/folder/'units.json', root/folder/scenario)
    options.enhanced_search = True
    whole = uninitialized_whole(d, scenarios, options)
    identity = fingerprint(whole.m)
    whole.initialize(solver.optimize(whole.m, whole.remaining()))
    before = whole.result.objective
    monkeypatch.setattr(solver, 'anneal', lambda *args: ([], {'bits': 1}))
    whole.diagnostics['joint_search']['stalled_rounds'] = 3
    whole.joint_step([], 1, 3)
    assert whole.diagnostics['joint_batches'][-1]['status'] in ('COMPLETE', 'LOCALLY_CLOSED')
    assert whole.result.validation['feasible'] and whole.result.objective == before
    assert fingerprint(whole.m) == identity


@pytest.mark.parametrize('legacy', [False, True])
def test_enhanced_resume_preserves_search_state_and_stream(tiny, options, monkeypatch, legacy):
    from copy import deepcopy
    from dataclasses import asdict
    whole = exchange_whole(tiny, options)
    whole.diagnostics.update(sa_calls=11, block_sa_calls=9, joint_sa_calls=2)
    whole.diagnostics['joint_search'].update(stalled_rounds=4, attempts=7)
    state = asdict(whole.result)
    state.update(charged_elapsed_seconds=2., resume_cursor=dict(outer=2, round=3), resume=dict(reason='test'))
    if legacy:
        for name in ('joint_search', 'joint_batches', 'joint_sa_calls', 'block_sa_calls'):
            del state['diagnostics'][name]
    original = deepcopy(state)
    recovered = uninitialized_whole(whole.d, whole.scenarios, options)
    monkeypatch.setattr(recovered.certifier, 'restore', lambda *a: None)
    recovered.restore(state)
    assert recovered.diagnostics['sa_calls'] == 11
    assert recovered.sampling_seed('joint') == whole.sampling_seed('joint')
    if legacy:
        assert recovered.diagnostics['joint_search']['migrated_legacy_checkpoint']
        assert recovered.diagnostics['joint_search']['attempts'] == 0
    else:
        assert recovered.diagnostics['joint_search'] == whole.diagnostics['joint_search']
    assert recovered.anchors() and recovered.result.validation['feasible']
    assert state['diagnostics'] == original['diagnostics']
