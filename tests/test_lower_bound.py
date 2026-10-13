from copy import deepcopy
from itertools import product
from types import SimpleNamespace as NS

import numpy as np
import pytest

from audit_helpers import quadratic_dual
from suc_benders.lower_bound import cover_cut, cut_pool, perspective_cut, proof_model, select_ids, verify_cut
from suc_benders.model import fingerprint, model, partition
from suc_benders.optimization import continuous_only
from suc_benders.validation import pack
from audit_helpers import verify_cut as verify_recourse_cut
from suc_benders.benders import dispatch, master


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


def test_capacity_covers_preserve_every_binary_capacity_feasible_point(tiny):
    d, s = tiny(n=4, T=1)
    d.pmax[:] = [10, 20, 35, 50]; s[0].load[:] = 56
    m = model(d, s)
    for bits in product((0, 1), repeat=4):
        subset = [i for i in range(4) if bits[i]]
        if d.pmax[subset].sum() >= 56:
            continue
        cut = cover_cut(m, 0, subset)
        verify_cut(m, cut)
        for u in product((0, 1), repeat=4):
            if d.pmax @ u >= 56:
                assert sum(u[i] for i in cut['complement']) >= 1


def test_cover_uses_strict_rational_condition(tiny):
    d, s = tiny(T=1)
    d.pmax[:] = [.1, .2]; s[0].load[:] = .3
    m = model(d, s)
    with pytest.raises(ValueError, match='strictly'):
        cover_cut(m, 0, [0, 1])
    assert cover_cut(m, 0, [1])['capacity'] == '1/5'


@pytest.mark.parametrize('change', ['model_fingerprint', 'p_coefficient', 'support', 'a'])
def test_forged_perspective_certificate_is_rejected(tiny, change):
    d, s = tiny(); m = model(d, s)
    cut = perspective_cut(m, 0, 0, 0, 20)
    cut[change] = 'wrong' if change == 'model_fingerprint' else cut[change]+1
    with pytest.raises(ValueError):
        proof_model(m, cuts=[cut])


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
    cuts = [cover_cut(m, t, [0]) for t in range(3)]
    proof = proof_model(m, cuts=cuts)
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
    proof = proof_model(m, perspective=False)
    expected = []
    for b in partition(m, 3)[0]:
        native, _ = master(b, [])
        expected += [kind for kind in native.kind if kind in ('and', 'capacity')]
    assert proof.labels[len(m.rhs):] == ['proof_'+kind for kind in expected]
    assert np.array_equal(proof.q[:len(m.q)], m.q)
    assert len(proof.time) == len(proof.obj)
    assert all(m.time[pair['v']] == m.time[pair['w']]-1 for pair in proof.and_lift)


def test_duplicate_cuts_do_not_change_model(tiny):
    d, s = tiny(); m = model(d, s)
    cuts = [perspective_cut(m, 0, 0, 0, 20), cover_cut(m, 0, [])]
    a = proof_model(m, cuts=cuts)
    b = proof_model(m, cuts=cuts*2)
    assert fingerprint(a) == fingerprint(b)


@pytest.mark.parametrize('value', [[], [0, 0], [True], [0., 1], [-1], [2], [0, 1, 2], '0'])
def test_bad_selections_fail_closed(value):
    with pytest.raises(ValueError, match='IDs'):
        select_ids(value, [dict(id=0, cut={}), dict(id=1, cut={})])


def test_pool_is_bounded_and_bound_to_model(tiny):
    d, s = tiny(T=3); m = model(d, s); proof = proof_model(m)
    point = np.zeros(len(proof.obj))
    point[m.ix.u] = .5; point[m.ix.p] = 12
    pool = cut_pool(m, proof, point, [0, 1, 2])
    assert len(pool) <= 32
    assert sum(row['cut']['kind'] == 'perspective' for row in pool) <= 16
    for row in pool:
        verify_cut(m, row['cut'])
    assert select_ids([pool[0]['id']], pool) == [pool[0]['cut']]
    changed = deepcopy(m); changed.rhs[0] += 1
    with pytest.raises(ValueError, match='match'):
        cut_pool(changed, proof, point, [0])


def test_cover_tampering_and_unknown_family_are_rejected(tiny):
    d, s = tiny(); m = model(d, s)
    cut = cover_cut(m, 0, []); cut['indices'] = [0]
    with pytest.raises(ValueError):
        verify_cut(m, cut)
    with pytest.raises(ValueError):
        verify_cut(m, dict(kind='invented'))






@pytest.mark.numerical
def test_perspective_feedback_cuts_bound_every_tiny_feasible_commitment(tiny):
    d, scenarios = tiny(T=1); m = model(d, scenarios)
    strengthened = proof_model(m, cuts=[perspective_cut(m, 0, 0, 0, 20.)])
    b = partition(m, 1)[0][0]; strong = partition(strengthened, 1)[0][0]
    b.audit_cuts = strong.audit_cuts = True
    points, certificates, calls = [], [], []
    with continuous_only(calls.append):
        for bits in product((0, 1), repeat=2):
            x = pack(d, np.asarray(bits).reshape(2, 1))
            exact = dispatch(b, x, 5, certified=True)
            relaxed = dispatch(strong, x, 5, certified=True)
            if not exact.feasible:
                continue
            assert relaxed.feasible and relaxed.cost <= exact.cost+1e-6
            points.append((x, exact.recourse))
            cut = relaxed.cut
            row = dict(alpha=cut.alpha, beta=cut.beta, optimality=cut.optimality, proof=cut.proof)
            verify_recourse_cut(strong, row); certificates.append(cut)
        assert len(points) == 3
        for x, cost in points:
            for cut in certificates:
                assert cut.alpha+cut.beta @ x <= cost+1e-6
    assert calls and all(call['integer_variables']==0 for call in calls)
