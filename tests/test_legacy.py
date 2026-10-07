"""Independent frozen D/17 code is the oracle, not the implementation under test."""
import importlib
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders.benders import dispatch, master
from suc_benders.model import model, partition
from suc_benders.qubo import qubo, quantize
from suc_benders.validation import commitment, pack, repair


@pytest.fixture(scope="module")
def legacy():
    package = ModuleType("d17_reference")
    package.__path__ = [str(Path(__file__).parent / "fixtures/d17")]
    sys.modules[package.__name__] = package
    data = ModuleType("d17_reference.data"); data.UCData = NS
    sys.modules[data.__name__] = data
    return NS(**{name: importlib.import_module(f"d17_reference.{name}")
                 for name in ("candidates", "paper_model", "master")})


def test_repair_matches_frozen_source(legacy, tiny):
    d, s = tiny(3, 7)
    d.min_up[:] = [3, 2, 1]; d.min_down[:] = [2, 3, 1]
    d.initial_on[:] = [1, 0, 0]; d.initial_duration[:] = [1, 1, 4]
    rng = np.random.default_rng(17)
    for raw in [np.full((3, 7), .5), np.zeros((3, 7)), *rng.uniform(-.5, 1.5, (60, 3, 7))]:
        old = legacy.candidates.repair_commitment(d, s, raw)
        new = repair(d, s, raw)
        assert (old is None) == (new is None)
        if old is not None:
            z = commitment(d, new)
            for name in ("u", "v", "w"):
                np.testing.assert_array_equal(getattr(z, name), getattr(old, name))


@pytest.mark.parametrize("budget", [1., .001])
def test_whole_cut_quantization_matches_frozen(legacy, budget):
    coefficients = {i: float(v) for i, v in enumerate(np.random.default_rng(3).normal(size=40))}
    old = legacy.master._quantize_inequality(coefficients, .123456789, budget)
    a, c, step = quantize(coefficients, .123456789, budget)
    assert a == old.coefficients and c == old.constant and step == old.step
    total_error = Fraction(float(.123456789)) - c*step
    total_error += sum(Fraction(v)-a[j]*step for j, v in coefficients.items())
    assert 0 <= total_error < Fraction(str(budget))


@pytest.mark.parametrize("minimum,with_cut", [(1, False), (1, True), (2, True)])
def test_sparse_qubo_matches_frozen_energy(legacy, tiny, options, minimum, with_cut):
    d, s = tiny(1, 2); d.min_up[:] = minimum
    b = partition(model(d, s), 2)[0][0]
    old_block = legacy.paper_model.Block(legacy.paper_model.IndexedUC(d, s))
    x = pack(d, np.ones((1, 2)))
    cut = dispatch(b, x, 10).cut
    cuts = [cut] if with_cut else []
    m, _ = master(b, cuts)
    m.ub[-1] = 400
    q = qubo(m, options, 400)
    keys = {}
    for name in ("u", "v", "w"):
        for i in range(d.n_units):
            for t in range(d.horizon):
                keys[int(getattr(b, name)[i, t])] = legacy.paper_model.key(name, i, t)
    old_cuts = [legacy.paper_model.Cut({keys[j]: float(v) for j, v in enumerate(cut.beta)}, cut.alpha)] if with_cut else []
    old = legacy.paper_model.CompactQubo(old_block, old_block.uc.linear, old_cuts, 0., 400., options)
    names = {j: legacy.paper_model.name(k) for j, k in keys.items()}
    for z, v, w in m.and_pairs:
        _, i, t, _ = keys[v]
        names[z] = f"and_{i}_{t}"
    order = [names[j] for j in q.free] + [f"z_{j}" for j in range(len(q.theta_ids))]
    slack = [name for name in old.names if name.startswith("_slack_")]
    slack.sort(key=lambda name: (0 if "reserve" in name else 1, *map(int, re.findall(r"\d+", name))))
    order.extend(slack)
    assert len(order) == q.Q.shape[0] == len(old.names)
    indices = [old.names.index(name) for name in order]
    matrix = old.matrix[np.ix_(indices, indices)]
    np.testing.assert_allclose(q.Q.toarray(), (matrix+matrix.T)/2, rtol=1e-11, atol=1e-7)
    assert q.offset == pytest.approx(float(old.model.make().offset), abs=1e-7)
    for bits in np.random.default_rng(17).integers(0, 2, (30, len(order))):
        assert float(bits @ q.Q @ bits + q.offset) == pytest.approx(
            float(bits @ matrix @ bits + old.model.make().offset), abs=2e-7)
