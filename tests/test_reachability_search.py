from itertools import combinations, product

import numpy as np
import pytest

from suc_benders.benders import master
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.neighborhood import joint_master, select_units
from suc_benders.qubo import fixed_binary
from suc_benders import solver
from suc_benders.validation import pack
from conftest import whole_day


def test_four_variable_propagation_preserves_every_binary_solution(tiny):
    d, scenarios = tiny(1, 3)
    d.pmin[:] = 40.; d.ramp_up[:] = 20.
    full = partition(model(d, scenarios), d.horizon)[0][0]
    guide = joint_master(full, [])
    fixed = fixed_binary(guide)
    assert np.array_equal(fixed[full.u], np.zeros((1, 3)))
    hard = np.asarray(guide.kind) == 'hard'
    A = guide.A[hard, :len(fixed)]
    for bits in product((0, 1), repeat=len(fixed)):
        residual = A @ bits-guide.rhs[hard]
        residual[guide.eq[hard]] = np.abs(residual[guide.eq[hard]])
        if np.max(residual, initial=0) <= 1e-9:
            known = np.isfinite(fixed)
            np.testing.assert_array_equal(np.asarray(bits)[known], fixed[known])


def test_rcuc_fixed_states_reach_every_block(options):
    d, scenarios = load_data('data/units.json', 'data/stochastic.json')
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    from time import perf_counter
    now = perf_counter()
    whole = solver.WholeDay(m, full, options, now+30, solver.Result(), now)
    movable = np.flatnonzero(np.isnan(whole.fixed[full.u]).any(axis=1))
    assert (movable+1).tolist() == [1, 4, 7, 9, 14, 17, 18, 19, 20]
    for j, b in enumerate(partition(m, 3)[0]):
        solver.BlockSearch(b, options, 1, j+1, whole)
        fixed = fixed_binary(master(b, [])[0])[:len(b.binary)]
        expected = whole.fixed[b.cols[b.binary]]
        known = np.isfinite(expected)
        np.testing.assert_array_equal(fixed[known], expected[known])


def test_combinations_cover_pairs_and_triples_without_starvation():
    movable = np.array([0, 3, 6, 8, 13, 16, 17, 18, 19])
    groups = [tuple(select_units(movable, k, 17)) for k in range(120)]
    expected = set(combinations(movable, 2)) | set(combinations(movable, 3))
    assert set(groups) == expected and len(set(groups)) == 120
    assert [len(g) for g in groups[:6]] == [2, 3, 2, 3, 2, 3]
    assert tuple(select_units(movable, 120, 17)) == groups[0]
    assert select_units(np.array([3]), 10, 17) == [3]
    assert select_units(np.array([3, 6]), 10, 17) == [3, 6]


def test_reachability_keeps_first_hour_stop_and_one_way_start(tiny):
    d, scenarios = tiny(2, 4)
    d.pmin[:] = 40.; d.pshut[:] = 45.; d.ramp_down[:] = 20.
    d.initial_on[0] = 1.; d.initial_output[0] = 10.
    full = partition(model(d, scenarios), 4)[0][0]
    fixed = fixed_binary(joint_master(full, []))
    assert np.isnan(fixed[full.w[0, 0]])  # Initial output permits this one shutdown.
    assert (fixed[full.w[:, 1:]] == 0).all()
    assert np.isnan(fixed[full.u[1]]).all()  # Can start but cannot subsequently stop.


def test_search_does_not_change_the_proof_master(tiny, options):
    from suc_benders.strengthened_certification import matrix_hash
    d, scenarios = tiny(2, 4)
    m = model(d, scenarios)
    full = partition(m, 4)[0][0]
    before = matrix_hash(master(full, [])[0])
    from time import perf_counter
    now = perf_counter()
    solver.WholeDay(m, full, options, now+30, solver.Result(), now)
    assert matrix_hash(master(full, [])[0]) == before


def test_cached_cut_is_reinstalled_without_dispatch(tiny, options, monkeypatch):
    options.enhanced_search = True
    whole = whole_day(tiny, options)
    x = pack(whole.d, whole.result.u)
    cut = whole.cache[x.astype(np.uint8).tobytes()]['cut']
    whole.diagnostics['joint_search']['recent_cuts'] = []
    count = len(whole.diagnostics['evaluations'])
    monkeypatch.setattr(solver, 'dispatch', lambda *a, **k: pytest.fail('cached dispatch'))
    assert whole.accept(x, 'joint_sa', {})
    assert whole.diagnostics['joint_search']['recent_cuts'] == [cut]
    assert len(whole.diagnostics['evaluations']) == count
    assert whole.diagnostics['joint_cuts_reused'] == 1


def test_raw_block_candidate_learns_before_repair(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    search = solver.BlockSearch(whole.full, options, 1, 1, whole)
    raw = np.zeros(len(whole.full.binary), dtype=np.uint8)
    monkeypatch.setattr(solver, 'closed', lambda *a, **k: False)
    monkeypatch.setattr(solver, 'anneal', lambda *a: ([raw], {'bits': 1}))
    search.step([pack(whole.d, whole.result.u)], 1)
    evaluated = search.cache[raw.tobytes()]
    assert not evaluated.feasible and evaluated.cut is not None
    assert any(c is evaluated.cut for c in search.cuts)
    assert any(row['source'] == 'sa_raw' and not row['feasible']
               for row in whole.diagnostics['local_evaluations'])
    assert whole.diagnostics['proposals'][-1]['raw_local_dispatch_completed']
    assert whole.diagnostics['proposals'][-1]['hamming_u'] > 0
