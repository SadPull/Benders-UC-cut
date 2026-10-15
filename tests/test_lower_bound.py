from itertools import product
from types import SimpleNamespace as NS

import numpy as np
import pytest

from audit_helpers import quadratic_dual
from suc_benders.strengthened_master import perspective_cut, proof_model, verify_cut
from suc_benders.model import fingerprint, model, partition
from suc_benders.benders import master


def lifted(m, proof, point):
    x = np.r_[point, np.zeros(len(proof.obj)-len(point))]
    for pair in proof.and_lift:
        x[pair['index']] = x[pair['v']]*x[pair['w']]
    for p, e in proof.eta.items():
        x[e] = m.q[p]*x[p]**2
    return x


def test_weighted_perspective_identity_and_zero_cost(tiny):
    d, _ = tiny(T=1)
    d.cost_a[1] = 0
    scenarios = [NS(name='a', probability=.25, load=d.base_load),
                 NS(name='b', probability=.75, load=d.base_load)]
    m = model(d, scenarios)
    for k in range(2):
        for p0 in (10., 32.5, 55.):
            cut = perspective_cut(m, 0, 0, k, p0)
            verify_cut(m, cut)
            for p in np.linspace(10, 55, 19):
                rhs = cut['p_coefficient']*p+cut['u_coefficient']
                assert cut['a']*p*p-rhs == pytest.approx(cut['a']*(p-p0)**2)
            assert cut['p_coefficient']*0+cut['u_coefficient']*0 == 0
    with pytest.raises(ValueError, match='positive'):
        perspective_cut(m, 1, 0, 0, 20)
    proof = proof_model(m)
    assert len(proof.eta) == 2
    assert not proof.q.any()
    assert np.isfinite(proof.ub).all()


@pytest.mark.parametrize('change', ['model_fingerprint', 'p_coefficient', 'support', 'a'])
def test_forged_perspective_certificate_is_rejected(tiny, change):
    d, s = tiny(); m = model(d, s)
    cut = perspective_cut(m, 0, 0, 0, 20)
    cut[change] = 'wrong' if change == 'model_fingerprint' else cut[change]+1
    with pytest.raises(ValueError):
        verify_cut(m, cut)


def test_proof_lift_preserves_physics_and_does_not_double_charge(tiny):
    d, s = tiny(T=3); m = model(d, s)
    before = fingerprint(m)
    proof = proof_model(m)
    assert (proof.A[:len(m.rhs), :len(m.obj)] != m.A).nnz == 0
    assert proof.A[:len(m.rhs), len(m.obj):].nnz == 0
    assert np.array_equal(proof.rhs[:len(m.rhs)], m.rhs)
    point = np.zeros(len(m.obj))
    point[m.ix.u[0]] = 1; point[m.ix.v[0, 0]] = 1
    point[m.ix.p[0, :, 0]] = 30
    x = lifted(m, proof, point)
    residual = proof.A @ x-proof.rhs
    residual[proof.eq] = abs(residual[proof.eq])
    assert residual.max() <= 1e-8
    assert proof.obj @ x+proof.q @ x**2 == pytest.approx(m.obj @ point+m.q @ point**2)
    assert fingerprint(m) == before
    assert fingerprint(proof) != before
    # Arbitrary valid multipliers independently bound this feasible lifted cost.
    rng = np.random.default_rng(7)
    for _ in range(10):
        prices = rng.random(len(proof.rhs))
        assert quadratic_dual(proof, prices) <= proof.obj @ x+1e-7


@pytest.mark.parametrize('min_up', [1, 2])
def test_all_tiny_feasible_grid_dispatches_survive_strengthening(tiny, min_up):
    from suc_benders.validation import pack
    d, s = tiny(T=3)
    d.pmax[:] = [20, 35]; d.pstart[:] = d.pmax; d.pshut[:] = d.pmax
    d.min_up[:] = min_up
    m = model(d, s)
    proof = proof_model(m)
    checked = 0
    for bits in product((0, 1), repeat=6):
        commitment = np.array(bits).reshape((2, 3))
        point = np.zeros(len(m.obj)); point[m.binary] = pack(d, commitment)
        for powers in product((0, 10, 20), repeat=3):
            point[m.ix.p[0, :, 0]] = powers
            point[m.ix.p[1, :, 0]] = 30-np.array(powers)
            residual = m.A @ point-m.rhs; residual[m.eq] = abs(residual[m.eq])
            if residual.max() > 1e-8:
                continue
            checked += 1
            x = lifted(m, proof, point)
            residual = proof.A @ x-proof.rhs; residual[proof.eq] = abs(residual[proof.eq])
            assert residual.max() <= 1e-8
            assert proof.obj @ x == pytest.approx(m.obj @ point+m.q @ point**2)
    assert checked > 0


def test_structural_rows_match_each_existing_block_master(tiny):
    d, s = tiny(T=6); m = model(d, s)
    proof = proof_model(m)
    expected = []
    for b in partition(m, d.horizon)[0]:
        native, _ = master(b, [])
        expected += [kind for kind in native.kind if kind in ('and', 'capacity')]
    assert [label for label in proof.labels[len(m.rhs):] if label.startswith('proof_')] == ['proof_'+kind for kind in expected]
    assert not proof.q.any()
    assert len(proof.time) == len(proof.obj)
    assert all(m.time[pair['v']] == m.time[pair['w']]-1 for pair in proof.and_lift)
