from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from suc_benders.anneal import anneal, initialize
from suc_benders.config import ROOT, load_options


@pytest.fixture(scope="module")
def authorized():
    o = load_options()
    if not o.kaiwu_user_id or not o.kaiwu_sdk_code:
        pytest.skip("Real Kaiwu tests need config.local.json authorization")
    initialize(o.kaiwu_user_id, o.kaiwu_sdk_code)


def small_qubo(matrix=None, energy_unit=1.):
    matrix = np.diag([-1., 2.]) if matrix is None else matrix
    return NS(Q=sparse.csr_matrix(matrix), offset=0., fixed=np.full(2, np.nan), free=np.arange(2),
              nx=2, energy_unit=energy_unit)


def test_auxiliary_spin_sign(options, monkeypatch):
    import kaiwu as kw
    monkeypatch.setattr(kw.classical.SimulatedAnnealingOptimizer, "solve", lambda *a: np.array([[1, -1, 1], [-1, 1, -1]]))
    rows, _ = anneal(small_qubo(), options, 10, 7)
    assert len(rows) == 2
    np.testing.assert_array_equal(rows[0], [1, 0])
    np.testing.assert_array_equal(rows[1], [1, 0])


def test_raw_sample_order_without_seed_injection(options, monkeypatch):
    import kaiwu as kw
    q = small_qubo()
    q.starts = np.array([[1.], [0.]])
    monkeypatch.setattr(kw.classical.SimulatedAnnealingOptimizer, "solve",
                        lambda *args: np.array([[-1, 1, 1], [1, -1, 1]]))
    rows, _ = anneal(q, options, 10, 17)
    np.testing.assert_array_equal(rows, [[0, 1], [1, 0]])


def test_energy_scaling(options, monkeypatch):
    import kaiwu as kw
    seen = []

    def optimizer(**kwargs):
        seen.append(kwargs)
        return NS(solve=lambda matrix: np.ones((1, 3)))

    monkeypatch.setattr(kw.classical, "SimulatedAnnealingOptimizer", optimizer)
    for factor in (1., 1e9):
        anneal(small_qubo(np.array([[-3., 2.], [2., 5.]]) * factor, .25 * factor), options, 10, 7)
    assert seen[0]["initial_temperature"] == pytest.approx(seen[1]["initial_temperature"])
    assert seen[0]["cutoff_temperature"] == pytest.approx(seen[1]["cutoff_temperature"])
    assert "process_num" not in seen[0]


@pytest.mark.integration
def test_real_small_qubo(authorized, options):
    a, _ = anneal(small_qubo(), options, 10, 7)
    b, _ = anneal(small_qubo(), options, 10, 7)
    np.testing.assert_array_equal(a, b)
    assert min(float(x @ np.diag([-1., 2.]) @ x) for x in a) == -1


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["deterministic", "stochastic"])
def test_real_rts_short_run(authorized, scenario):
    from run_suc import run_suc
    o = {"scenario_file": str(ROOT / f"data/{scenario}.json"), "max_outer": 6, "max_inner": 2,
         "time_limit": 120, "sa_reads": 4, "sa_sweeps": 2, "sa_cooling": .8,
         "sa_temperature": 10, "verbose": False, "llm_enabled": False}
    before = np.random.get_state()
    a = run_suc(o)
    after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    assert a.status in ("MAX_ITERATIONS", "DUAL_CONVERGED", "TIME_LIMIT"), a
    assert a.validation["feasible"] and np.isfinite(a.objective)
    assert a.lower_bound <= a.objective + 1e-5
    assert a.history and a.history[0]["sampling"]["bits"] > 0
    assert a.diagnostics["sa_calls"] > 0 and a.llm_calls == 0
    assert {row["block"] for row in a.diagnostics["blocks"] if row["visits"]} == set(range(1, 9))
    assert all(row["sa_calls"] > 0 or row["converged"] for row in a.diagnostics["blocks"])
    assert a.diagnostics["coordination_count"] == 6 and len(a.outer_history) == 6
    assert a.diagnostics["complete_rounds"] >= 1
    assert all(row["seed"] == 7 for row in a.diagnostics["sampling"])
    assert a.diagnostics["timings"]["qubo"] > 0
    print(f"\n{scenario}: status={a.status} cost={a.objective:.9f} LB={a.lower_bound:.9f} gap={a.relative_gap:.6%} residual={a.validation['max_residual_mw']:.3g} seconds={a.elapsed_seconds:.3f}")
