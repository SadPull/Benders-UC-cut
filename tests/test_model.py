import unittest
from pathlib import Path
import numpy as np

from suc_benders import Scenario, load_suc_data
from suc_benders.common import commitment_is_feasible
from suc_benders.master import QuantumConfig
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.subproblem import solve_scenario, cut_value


ROOT = Path(__file__).resolve().parents[1]


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

    def test_all_on_is_discrete_feasible(self):
        n, h = self.units.n_units, self.units.horizon
        u = np.ones((n, h), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        v[:, 0] = 1 - self.units.initial_on.astype(int)
        self.assertTrue(commitment_is_feasible(self.units, self.scenarios, u, v, w))

    def test_bad_shape_and_binary_rejected(self):
        n, h = self.units.n_units, self.units.horizon
        u = np.ones((n, h), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        self.assertFalse(commitment_is_feasible(self.units, self.scenarios, u[:-1], v, w))
        u[0, 0] = 2
        self.assertFalse(commitment_is_feasible(self.units, self.scenarios, u, v, w))


class SubproblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")
        n, h = cls.units.n_units, cls.units.horizon
        cls.u = np.ones((n, h), dtype=int)
        cls.v = np.zeros_like(cls.u)
        cls.w = np.zeros_like(cls.u)
        cls.v[:, 0] = 1 - cls.units.initial_on.astype(int)

    def test_feasible_scenario_dispatch_and_cut(self):
        scenario = self.scenarios[1]
        result = solve_scenario(self.units, scenario, self.u, self.v, self.w)
        self.assertTrue(result.feasible)
        self.assertEqual(result.status, "OPTIMAL")
        np.testing.assert_allclose(np.sum(result.p, axis=0), scenario.load, atol=1e-5)
        self.assertGreaterEqual(np.sum(result.r, axis=0).min(), self.units.reserve_fraction * scenario.load.min() - 1e-5)
        self.assertAlmostEqual(cut_value(result, self.u, self.v, self.w), result.objective, places=5)

    def test_infeasible_scenario_returns_feasibility_cut(self):
        scenario = Scenario("impossible", 1.0, np.full(self.units.horizon, self.units.pmax.sum() + 1.0))
        result = solve_scenario(self.units, scenario, self.u, self.v, self.w)
        self.assertFalse(result.feasible)
        self.assertTrue(np.isfinite(result.cut_alpha))
        self.assertTrue(np.all(np.isfinite(result.cut_beta_u)))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

    def test_two_inner_iteration_rts24_run(self):
        config = BendersConfig(
            max_iterations=2,
            max_inner_iterations=2,
            quantum=QuantumConfig(initial_temperature=20, alpha=0.9, cutoff_temperature=0.1, iterations_per_temperature=2, candidate_pool_size=4),
        )
        result = solve_benders(self.units, self.scenarios, config)
        self.assertIn(result.status, {"MAX_ITERATIONS", "INNER_NOT_CONVERGED", "DUAL_CONVERGED", "TIME_LIMIT"})
        self.assertTrue(np.isfinite(result.objective))
        self.assertTrue(commitment_is_feasible(self.units, self.scenarios, result.u, result.v, result.w))
        self.assertEqual(len(result.scenarios), 3)
        self.assertGreaterEqual(result.optimality_cuts + result.feasibility_cuts, 1)


if __name__ == "__main__":
    unittest.main()
