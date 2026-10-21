from types import SimpleNamespace as NS

import numpy as np
import pytest

from conftest import make_whole
from suc_benders import lagrangian
from suc_benders.model import model, partition, priced, relaxed_rows, search_multipliers
from suc_benders.optimization import optimize


# 带冷启动、最小开停机 2 小时、爬坡 20 MW 的可行小算例；初始停机已满足最小停机时间。
def cold(tiny, n=2, T=6):
    d, scenarios = tiny(n, T)
    d.cold_startup_cost = d.startup_cost * 3; d.cold_start_hours = np.full(n, 1.)
    d.min_up[:] = 2; d.min_down[:] = 2; d.ramp_up[:] = 20; d.ramp_down[:] = 20
    d.initial_duration[:] = 5
    scenarios[0].load[:] = [30, 50, 70, 40, 60, 30][:T]
    return d, scenarios


def multipliers(m, cross, seed):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 5, len(cross))
    lam[m.eq[cross]] = rng.uniform(-5, 5, m.eq[cross].sum())
    return lam


@pytest.mark.parametrize('hours', [2, 3])
def test_priced_blocks_sum_to_full_lagrangian(tiny, hours):
    d, scenarios = cold(tiny)
    m = model(d, scenarios)
    blocks, cross = partition(m, hours)
    assert len(blocks) > 1 and len(cross)
    rng = np.random.default_rng(3)
    for seed in range(3):
        lam = multipliers(m, cross, seed)
        x = rng.uniform(m.lb, np.minimum(m.ub, 60))
        full = m.obj @ x + m.q @ x**2 + lam @ (m.A[cross] @ x - m.rhs[cross])
        pieces = [priced(b, lam) for b in blocks]
        total = sum(p.obj @ x[p.cols] + p.q @ x[p.cols]**2 + p.constant for p in pieces)
        assert total == pytest.approx(full, rel=1e-12, abs=1e-9)
        # 未定价的块与原目标一致，priced 不修改输入块。
        assert all(np.array_equal(b.obj, m.obj[b.cols]) and b.constant == 0. for b in blocks)


def test_priced_rejects_negative_inequality_multipliers(tiny):
    m = model(*tiny(2, 6))
    blocks, cross = partition(m, 3)
    lam = np.zeros(len(cross)); lam[np.flatnonzero(~m.eq[cross])[0]] = -1e-3
    with pytest.raises(ValueError, match='nonnegative'):
        priced(blocks[0], lam)
    with pytest.raises(ValueError, match='match'):
        priced(blocks[0], np.zeros(len(cross) + 1))


def test_relaxed_rows_are_exactly_the_cross_rows_with_continuous_columns(tiny):
    d, scenarios = cold(tiny)
    m = model(d, scenarios)
    _, cross = partition(m, 3)
    relaxed = relaxed_rows(m, cross)
    labels = np.asarray(m.labels)[cross]
    assert set(labels[relaxed]) == {'ramp_up', 'ramp_down', 'cold_start', 'shutdown_capacity'}
    assert set(labels[~relaxed]) <= {'transition', 'min_up', 'min_down'}
    lam = multipliers(m, cross, 0)
    search = search_multipliers(m, cross, lam)
    assert np.array_equal(search[relaxed], lam[relaxed]) and not search[~relaxed].any()


def test_partition_rejects_rows_without_coefficients(tiny):
    m = model(*tiny(2, 6))
    m.A = m.A.tolil(); m.A[0, :] = 0; m.A = m.A.tocsr(); m.A.eliminate_zeros()
    with pytest.raises(ValueError, match='without coefficients'):
        partition(m, 3)


# 冷启动小算例的分块、跨块行、全天固定位、已校验 incumbent 与 LP 松弛。
def decomposed(tiny, options, hours=3):
    d, scenarios = cold(tiny)
    whole = make_whole(d, scenarios, options)
    blocks, cross = partition(whole.m, hours)
    fixed = [whole.fixed[b.cols[b.binary]] for b in blocks]
    relaxation = optimize(whole.m, 60)
    return whole, blocks, cross, fixed, relaxation


def evaluate_all(blocks, fixed, lam, whole):
    return [lagrangian.oracle(b, f, lam, 60, whole.best.point) for b, f in zip(blocks, fixed)]


def test_lagrangian_never_exceeds_full_optimum(tiny, options):
    whole, blocks, cross, fixed, relaxation = decomposed(tiny, options)
    optimum = optimize(whole.m, 60, integral=True).objval
    lp = relaxation.prices[cross]
    random = np.random.default_rng(5).uniform(0, 20, len(cross))
    random[whole.m.eq[cross]] -= 10
    for lam in (np.zeros(len(cross)), lp, 3*lp, random):
        results = evaluate_all(blocks, fixed, lam, whole)
        assert all(r.exact for r in results)
        bound, value = sum(r.bound for r in results), sum(r.value for r in results)
        assert bound <= value + 1e-6 and value <= optimum + 1e-4 * max(1, abs(optimum))
    # LP 对偶处的拉格朗日值不低于 LP 松弛界（块子问题保留了整数性和固定位）。
    assert sum(r.value for r in evaluate_all(blocks, fixed, lp, whole)) >= relaxation.lower - 1e-4


def test_plane_is_tight_where_generated_and_overestimates_elsewhere(tiny, options):
    whole, blocks, cross, fixed, relaxation = decomposed(tiny, options)
    first, second = relaxation.prices[cross], 2*relaxation.prices[cross] + 1.
    second[whole.m.eq[cross]] = -3.
    for b, f in zip(blocks, fixed):
        r = lagrangian.oracle(b, f, first, 60, whole.best.point)
        a, g = r.plane
        assert a + g @ first == pytest.approx(r.value, rel=1e-9, abs=1e-9)
        other = lagrangian.oracle(b, f, second, 60, whole.best.point)
        assert a + g @ second >= other.value - 1e-6 * max(1, abs(other.value))


def test_oracle_point_respects_fixed_bits(tiny, options):
    whole, blocks, cross, fixed, relaxation = decomposed(tiny, options)
    b = blocks[1]
    forced = fixed[1].copy()
    free = np.flatnonzero(np.isnan(forced) & np.isin(np.arange(len(forced)), b.u.ravel()))
    # 只固定 incumbent（全开）满足的位，并用高价让无约束时更想关机。
    forced[free[:2]] = 1.
    lam = relaxation.prices[cross]
    r = lagrangian.oracle(b, forced, lam, 60, whole.best.point)
    assert np.array_equal(r.point[b.binary][free[:2]], [1., 1.])
    assert np.array_equal(r.point[b.binary][np.isfinite(fixed[1])], fixed[1][np.isfinite(fixed[1])])


def test_bundle_with_real_oracle_improves_and_stays_below_optimum(tiny, options):
    whole, blocks, cross, fixed, relaxation = decomposed(tiny, options, hours=2)
    optimum = optimize(whole.m, 60, integral=True).objval
    state = lagrangian.start(len(blocks), whole.m.eq[cross], relaxation.prices[cross])
    lagrangian.maximize(state, lambda j, lam, s: lagrangian.oracle(blocks[j], fixed[j], lam, s, whole.best.point),
                        iterations=15, remaining=lambda: 60.)
    assert state.status in ("CONVERGED", "ITERATION_LIMIT"), state.status
    assert state.initial_bound <= state.bound <= optimum + 1e-6
    centers = [state.initial_value] + [row["center_value"] for row in state.history]
    assert all(b >= a for a, b in zip(centers, centers[1:]))
    record = lagrangian.record(state)
    assert record["iterations"] == state.iterations and record["planes"] >= len(blocks)


# 合成凹对偶：每块由若干 (截距, 斜率) 线性片取最小，预言机返回取到最小值的片。
def synthetic(pieces, calls=None, status=None):
    def evaluate(j, lam, seconds):
        if calls is not None:
            calls.append(lam.copy())
        if status is not None and len(calls) > 1:
            return NS(status=status, exact=False, value=np.inf, bound=-np.inf, plane=None)
        a, g = min(pieces[j], key=lambda piece: piece[0] + np.asarray(piece[1]) @ lam)
        value = a + np.asarray(g) @ lam
        return NS(status="OPTIMAL", exact=True, value=value, bound=value, plane=(a, np.asarray(g, float)))
    return evaluate


def test_bundle_reaches_max_of_piecewise_concave_dual():
    state = lagrangian.start(1, np.array([True]), np.zeros(1))
    lagrangian.maximize(state, synthetic([[(10., [1.]), (30., [-1.])]]), iterations=30, remaining=lambda: 10.)
    assert state.status == "CONVERGED"
    assert state.center[0] == pytest.approx(10.) and state.center_value == pytest.approx(20.)
    assert state.serious > 0 and all(row["center_value"] <= 20. + 1e-9 for row in state.history)


def test_binding_box_does_not_stop_early():
    state = lagrangian.start(1, np.array([False]), np.zeros(1))
    pieces = [[(1e6, [1e-2]), (1e6 + 1e4, [-1e-2])]]
    lagrangian.maximize(state, synthetic(pieces), iterations=60, remaining=lambda: 10.)
    # 盒半径 1 时预测增益 0.01 低于容差，但盒约束起作用：必须扩盒而不是在 λ0 判收敛。
    assert state.expansions > 0 and state.status == "CONVERGED"
    assert state.center_value >= 1e6 + 5e3 - 10.


def test_degraded_oracle_and_master_keep_the_center(monkeypatch):
    pieces = [[(10., [1.]), (30., [-1.])]]
    calls = []
    state = lagrangian.start(1, np.array([True]), np.array([2.]))
    lagrangian.maximize(state, synthetic(pieces, calls, status="NUMERIC"), iterations=10, remaining=lambda: 10.)
    assert state.status == "ORACLE_NUMERIC" and state.oracle_failures == {"NUMERIC": 1}
    assert state.center.tolist() == [2.] and state.center_value == 12.
    monkeypatch.setattr(lagrangian, "optimize", lambda m, seconds: NS(status="NUMERIC"))
    state = lagrangian.start(1, np.array([True]), np.array([2.]))
    lagrangian.maximize(state, synthetic(pieces), iterations=10, remaining=lambda: 10.)
    assert state.status == "MASTER_NUMERIC" and state.master_failures == {"NUMERIC": 1}
    assert state.center.tolist() == [2.] and lagrangian.record(state)["multiplier_change"] == 0.


def test_bundle_limits_and_interruptions_have_explicit_statuses():
    pieces = [[(10., [1.]), (30., [-1.])]]
    state = lagrangian.maximize(lagrangian.start(1, np.array([True]), np.zeros(1)), synthetic(pieces),
                                iterations=0, remaining=lambda: 10.)
    assert state.status == "NOT_RUN" and state.message == "ZERO_ITERATIONS"
    state = lagrangian.maximize(lagrangian.start(1, np.array([True]), np.zeros(1)), synthetic(pieces),
                                iterations=5, remaining=lambda: 0.)
    assert state.status == "TIME_LIMIT" and state.oracle_calls == 0
    state = lagrangian.maximize(lagrangian.start(1, np.array([True]), np.zeros(1)), synthetic(pieces),
                                iterations=2, remaining=lambda: 10.)
    assert state.status == "ITERATION_LIMIT" and state.iterations == 2
    polls = iter([False, True])
    state = lagrangian.maximize(lagrangian.start(1, np.array([True]), np.zeros(1)), synthetic(pieces),
                                iterations=5, remaining=lambda: 10., poll=lambda: next(polls, True))
    assert state.status == "INTERRUPTED" and state.iterations == 1


def test_plane_below_block_bound_is_rejected():
    def wrong(j, lam, seconds):
        return NS(status="OPTIMAL", exact=True, value=5., bound=5., plane=(4., np.zeros(1)))
    with pytest.raises(RuntimeError, match="below the block bound"):
        lagrangian.maximize(lagrangian.start(1, np.array([True]), np.zeros(1)), wrong,
                            iterations=5, remaining=lambda: 10.)
