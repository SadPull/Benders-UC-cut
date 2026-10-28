from itertools import product

import numpy as np
import pytest

import offline_pilot as pilot
import repair_headroom as rh
from suc_benders import search as search_core
from suc_benders.model import model, partition
from suc_benders.validation import commitment, commitment_hash, pack, repair


# 2 台机组：最小开机 2 小时、费用不同，使最小改动修复在 Hamming 并列时须按真实成本取舍；可选冷启动费。
def two_units(tiny, cold=False):
    d, scenarios = tiny(2, 4)
    d.min_up = np.array([2., 2.])
    d.cost_b, d.cost_c = np.array([2., 3.]), np.array([5., 1.])
    if cold:
        d.cold_startup_cost, d.cold_start_hours = np.array([8., 9.]), np.array([1., 1.])
    m = model(d, scenarios)
    return m, partition(m, d.horizon)[0][0]


def feasible_costs(m, full, shape):
    rows = []
    for bits in product((0., 1.), repeat=shape[0]*shape[1]):
        u = np.asarray(bits).reshape(shape)
        result = rh.evaluate_u(m, full, u)
        if result['feasible']:
            rows.append((u, result['objective']))
    return rows


def brute_force(m, full, raw):
    feasible = [(int(np.count_nonzero(u != raw)), c) for u, c in feasible_costs(m, full, raw.shape)]
    h = min(h for h, _ in feasible)
    return h, min(c for g, c in feasible if g == h)


def test_bits_round_trip_including_padding():
    rng = np.random.default_rng(3)
    for shape in ((3, 5), (2, 4), (54, 24)):
        u = rng.integers(0, 2, shape).astype(float)
        assert np.array_equal(rh.bits_to_u(rh.u_to_bits(u), shape), u)


def test_hamming_row_counts_exactly_and_touches_only_u_columns(tiny):
    m, _ = two_units(tiny, cold=True)
    assert hasattr(m.ix, 'cold_start')
    rng = np.random.default_rng(5)
    raw = rng.integers(0, 2, (2, 4)).astype(float)
    coef, offset = rh.hamming_coefficients(m, raw)
    assert set(np.flatnonzero(coef)) <= set(m.ix.u.ravel())
    for _ in range(20):
        u = rng.integers(0, 2, (2, 4)).astype(float)
        x = rng.random(len(m.obj)); x[:3*u.size] = pack(m.d, u)
        assert coef @ x + offset == pytest.approx(np.count_nonzero(u != raw))


@pytest.mark.parametrize('cold', [False, True])
def test_minimal_change_matches_brute_force(tiny, cold):
    m, full = two_units(tiny, cold)
    rng = np.random.default_rng(11)
    raws = [np.zeros((2, 4)), np.array([[1., 0, 1, 0], [0, 1, 0, 0]]), *rng.integers(0, 2, (3, 2, 4)).astype(float)]
    for raw in raws:
        h, cost = brute_force(m, full, raw)
        p0 = rh.minimal_change(m, full, raw)
        assert p0['phase1_status'] == p0['phase2_status'] == 'OPTIMAL'
        assert p0['h_star'] == p0['hamming'] == h and p0['feasible']
        assert p0['objective'] == pytest.approx(cost, rel=1e-6, abs=1e-6)
        assert abs(p0['mip_objective']-p0['objective']) <= 1e-5


def test_feasible_raw_needs_no_change_and_matches_current_repair(tiny):
    d, scenarios = tiny(2, 4)
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    raw = np.ones((2, 4))
    current, p0 = rh.current_repair(m, full, raw), rh.minimal_change(m, full, raw)
    assert current['hamming'] == 0 and current['feasible'] and p0['h_star'] == 0
    assert p0['objective'] == pytest.approx(current['objective'], rel=1e-9)


def test_controls_bracket_the_incumbent(tiny):
    m, full = two_units(tiny)
    anchor = np.ones((2, 4))
    incumbent = rh.evaluate_u(m, full, anchor)['objective']
    assert rh.ball(m, full, anchor, 0)['objective'] == pytest.approx(incumbent, rel=1e-9)
    # 块覆盖全天时块内上限就是全局最优；块外全部固定时只能回到现任方案。
    whole_day = rh.block_ceiling(m, full, anchor, np.arange(4))
    optimum = min(c for _, c in feasible_costs(m, full, anchor.shape))
    assert whole_day['feasible'] and whole_day['objective'] == pytest.approx(optimum, rel=1e-6)
    assert rh.block_ceiling(m, full, anchor, np.arange(0))['objective'] == pytest.approx(incumbent, rel=1e-9)


def trial(d, scenarios, options, logged):
    kept = {}
    keep = lambda whole, event: kept.setdefault('whole', whole)
    if logged:
        with rh.log_proposals() as holder:
            row = pilot.run_trial('tiny', 'SA', 5, 30., {}, rounds=2, data=(d, scenarios), o=options, progress_callback=keep)
    else:
        holder = None
        row = pilot.run_trial('tiny', 'SA', 5, 30., {}, rounds=2, data=(d, scenarios), o=options, progress_callback=keep)
    return row, kept['whole'], holder


def test_logging_wrapper_changes_nothing_and_logs_reproducible_samples(tiny, options):
    d, scenarios = tiny(3, 6)
    original = search_core.WholeDay.prepare
    plain, plain_whole, _ = trial(d, scenarios, options, False)
    logged, logged_whole, holder = trial(d, scenarios, options, True)
    assert search_core.WholeDay.prepare is original
    assert plain['final_u'] == logged['final_u'] and plain['sa_calls'] == logged['sa_calls'] > 0
    assert ([(r['proposal_id'], r['outcome']) for r in plain_whole.diagnostics['proposals']]
            == [(r['proposal_id'], r['outcome']) for r in logged_whole.diagnostics['proposals']])
    assert ([(e['source'], e['commitment_hash'], e['objective']) for e in plain_whole.diagnostics['evaluations']]
            == [(e['source'], e['commitment_hash'], e['objective']) for e in logged_whole.diagnostics['evaluations']])
    rows = rh.collect(logged_whole)
    assert rows and {r['origin'] for r in rows} >= {'sa'}
    assert all(r['incumbent_id'] in holder.incumbents for r in rows)
    repaired = [r for r in rows if r['commitment_id'] is not None]
    assert repaired
    for r in repaired:
        raw = rh.bits_to_u(r['raw_bits'], (3, 6))
        z = commitment(d, repair(d, scenarios, raw))
        assert commitment_hash(z) == r['commitment_id'] and int(np.count_nonzero(z.u != raw)) == r['hamming_u']
    for r in rows:
        if r['origin'] == 'sa':
            assert r['anchor_cost'] == pytest.approx(r['incumbent_objective'], rel=1e-12)


def synthetic_rows():
    anchor = np.zeros((2, 4))
    incumbents = {'a': dict(bits=rh.u_to_bits(anchor), objective=10.)}
    rows, order, index = [], 0, 0
    # 每个样本只开一个不同的位，互不相同且都不等于现任方案；样本 1-0 出现两次。
    for block, count in ((1, 6), (2, 2)):
        for k in range(count):
            u = anchor.copy(); u.ravel()[index] = 1; index += 1
            for _ in range(1+(block == 1 and k == 0)):
                rows.append(dict(origin='sa', block=block, raw_sha256=f'{block}-{k}', raw_bits=rh.u_to_bits(u),
                                 incumbent_id='a', order=order)); order += 1
    rows.append(dict(origin='sa', block=2, raw_sha256='same', raw_bits=rh.u_to_bits(anchor), incumbent_id='a', order=order))
    rows.append(dict(origin='pool', block=1, raw_sha256='pool', raw_bits=rh.u_to_bits(anchor+1), incumbent_id='a', order=order+1))
    return rows, incumbents


def test_select_is_deterministic_proportional_and_drops_the_incumbent():
    rows, incumbents = synthetic_rows()
    chosen, info = rh.select(rows, incumbents, (2, 4), 4, 17)
    again, _ = rh.select(rows, incumbents, (2, 4), 4, 17)
    assert [r['raw_sha256'] for r in chosen] == [r['raw_sha256'] for r in again]
    assert info['identical_to_incumbent'] == 1 and info['eligible'] == 8 and info['per_block'] == {'1': 3, '2': 1}
    assert 'same' not in {r['raw_sha256'] for r in chosen} and len(chosen) == 4
    assert next(r for r in rh.select(rows, incumbents, (2, 4), 8, 17)[0] if r['raw_sha256'] == '1-0')['multiplicity'] == 2
    assert [r['raw_sha256'] for r in rh.select(rows, incumbents, (2, 4), 5, 17, origin='pool')[0]] == ['pool']


CRITERIA = dict(seeds=5, pass_seeds=3, pass_rate=.01, fail_seeds=1, max_unresolved=.10, min_replayed=60)


def classes(improving, total=100, unresolved=0):
    return [['improving']*k + ['unresolved']*unresolved + ['non_improving']*(total-k-unresolved) for k in improving]


def test_gate_boundaries():
    gate = lambda groups: rh.gate(dict(enumerate(groups)), CRITERIA)['decision']
    assert gate(classes([2, 1, 2, 0, 0])) == 'PASS'                # 3/5 个种子，合并恰好 1%
    assert gate(classes([1, 1, 2, 0, 0])) == 'INCONCLUSIVE'        # 3/5 个种子，但只有 0.8%
    assert gate(classes([5, 0, 0, 0, 0])) == 'FAIL'                # 1/5 个种子
    assert gate(classes([5, 0, 0, 0, 0], unresolved=11)) == 'INCONCLUSIVE'
    assert gate(classes([0, 0, 0, 0, 0], total=59)) == 'INCONCLUSIVE'
    assert gate(classes([0, 0, 0, 0])) == 'INCONCLUSIVE'           # 少一个种子不能判 FAIL


def test_classify_handles_time_limits():
    incumbent = 100.
    assert rh.classify(dict(phase1_status='TIME_LIMIT', objective=50.), incumbent) == 'unresolved'
    assert rh.classify(dict(phase1_status='OPTIMAL', phase2_status='TIME_LIMIT', objective=99.), incumbent) == 'improving'
    assert rh.classify(dict(phase1_status='OPTIMAL', phase2_status='OPTIMAL', objective=100.), incumbent) == 'non_improving'
    assert rh.classify(dict(phase1_status='OPTIMAL', phase2_status='TIME_LIMIT', objective=101., bound=99.99995),
                       incumbent) == 'non_improving'
    assert rh.classify(dict(phase1_status='OPTIMAL', phase2_status='TIME_LIMIT', objective=101., bound=99.),
                       incumbent) == 'unresolved'
    assert rh.classify(dict(phase1_status='OPTIMAL', phase2_status='OPTIMAL', objective=None), incumbent) == 'unresolved'
