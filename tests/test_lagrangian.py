import numpy as np
import pytest

from suc_benders.model import model, partition, priced, relaxed_rows, search_multipliers


def cold(tiny, n=2, T=6):
    d, scenarios = tiny(n, T)
    d.cold_startup_cost = d.startup_cost * 3; d.cold_start_hours = np.full(n, 1.)
    d.min_up[:] = 2; d.min_down[:] = 2; d.ramp_up[:] = 20; d.ramp_down[:] = 20
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
