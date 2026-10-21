import numpy as np
import pytest

from suc_benders import strengthened_master as strong
from suc_benders.benders import master
from suc_benders.model import fingerprint, model, partition


def proof_base(tiny):
    d, scenarios = tiny(2, 4)
    physical = model(d, scenarios)
    root, _ = master(partition(physical, d.horizon)[0][0], [])
    root = strong.add_temporal(root, physical)
    return physical, strong.build(physical, root, len(root.obj)-1)


def fractional_point(physical, base):
    # u=0.5 和 p=20 让每个透视项都有缺口；eta 留在 0。
    point = np.zeros(len(base.obj))
    point[physical.ix.u.ravel()] = .5
    point[physical.ix.p.ravel()] = 20.
    return point


def test_perspective_round_fingerprints_the_model_once_per_call(tiny, monkeypatch):
    physical, base = proof_base(tiny)
    identity = fingerprint(physical)
    calls = []
    monkeypatch.setattr(strong, 'fingerprint', lambda m: calls.append(m) or identity)
    gap, cuts = strong.perspective_batch(physical, base, fractional_point(physical, base), .01)
    refined = strong.add_perspectives(base, physical, cuts)
    assert gap > 0 and len(cuts) == len(base.eta) and len(calls) == 2
    assert all(cut['model_fingerprint'] == identity for cut in cuts)
    assert refined.A.shape[0] == base.A.shape[0] + len(cuts)


def test_tampered_perspective_cut_is_rejected(tiny):
    physical, base = proof_base(tiny)
    _, cuts = strong.perspective_batch(physical, base, fractional_point(physical, base), .01)
    cuts[0] = dict(cuts[0], p_coefficient=cuts[0]['p_coefficient']*2)
    with pytest.raises(ValueError, match='changed'):
        strong.add_perspectives(base, physical, cuts)
