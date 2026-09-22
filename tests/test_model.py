import unittest
from pathlib import Path
import numpy as np

from suc_benders import DataError, Scenario, load_suc_data
from suc_benders.common import commitment_is_feasible
from suc_benders.master import BendersCut, QuantumConfig, QuantumMaster, theta_bit_width
from suc_benders.quantum import solve_qubo
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.subproblem import solve_scenario, cut_value


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOAD = [700, 750, 850, 950, 1000, 1100, 1150, 1200, 1300, 1400, 1450, 1500,
                 1400, 1300, 1200, 1050, 1000, 1100, 1200, 1400, 1300, 1100, 900, 800]


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_canonical_shape_and_load(self):
        self.assertEqual(self.units.n_units, 10)
        self.assertEqual(self.units.horizon, 24)
        self.assertEqual(self.units.base_load.tolist(), EXPECTED_LOAD)
        self.assertEqual(self.units.pmax[:2].tolist(), [455.0, 455.0])
        self.assertEqual(self.units.pmin[-1], 10.0)
        self.assertEqual(self.units.min_up[:2].tolist(), [8, 8])
        self.assertEqual(self.units.initial_on.astype(int).tolist(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_default_scenarios(self):
        self.assertEqual([s.name for s in self.scenarios], ["low", "base", "high"])
        self.assertEqual([s.probability for s in self.scenarios], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(self.scenarios[0].load, 0.9 * np.asarray(EXPECTED_LOAD))
        np.testing.assert_allclose(self.scenarios[2].load, 1.05 * np.asarray(EXPECTED_LOAD))

    def test_scenario_file_and_validation(self):
        units, scenarios = load_suc_data(ROOT / "data/suc_10x24.json", ROOT / "data/scenarios.json")
        self.assertEqual([s.name for s in scenarios], ["low", "base", "high"])

        bad = ROOT / "data/_bad_scenarios.json"
        original = (ROOT / "data/scenarios.json").read_text(encoding="utf-8")
        try:
            bad.write_text(original.replace('"probability": 0.25, "load_factor": 0.9', '"probability": 0.24, "load_factor": 0.9'), encoding="utf-8")
            with self.assertRaises(DataError):
                load_suc_data(ROOT / "data/suc_10x24.json", bad)
            bad.write_text(original.replace('"load_factor": 0.9', '"load": [1, 2]'), encoding="utf-8")
            with self.assertRaises(DataError):
                load_suc_data(ROOT / "data/suc_10x24.json", bad)
            bad.write_text(original.replace('"load_factor": 0.9', '"load_factor": 0.9, "load": [1]'), encoding="utf-8")
            with self.assertRaises(DataError):
                load_suc_data(ROOT / "data/suc_10x24.json", bad)
        finally:
            bad.unlink(missing_ok=True)


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

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


class QuantumMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_theta_width_and_variable_count(self):
        self.assertEqual(theta_bit_width(1023, 1), 10)
        self.assertEqual(theta_bit_width(1024, 1), 11)
        master = QuantumMaster(self.units, self.scenarios, [], theta_cap=1023)
        self.assertEqual(master.variable_count(), 730)
        self.assertEqual(master.matrix().shape, (730, 730))

    def test_cut_changes_matrix_and_decode_roundtrip(self):
        shape = (self.units.n_units, self.units.horizon)
        beta = np.ones(shape)
        cut = BendersCut(-1.0, beta, -beta, beta)
        base = QuantumMaster(self.units, self.scenarios, [], theta_cap=1024)
        with_cut = QuantumMaster(self.units, self.scenarios, [cut], theta_cap=1024)
        self.assertEqual(with_cut.variable_count(), base.variable_count())
        self.assertFalse(np.array_equal(base.matrix(), with_cut.matrix()))
        binary = np.zeros(base.variable_count(), dtype=int)
        candidate = with_cut.decode(binary)
        self.assertTrue(np.all(candidate.u == 0))
        self.assertTrue(np.all(candidate.v == 0))
        self.assertTrue(np.all(candidate.w == 0))
        self.assertEqual(candidate.theta, 0.0)

    def test_kaiwu_local_sa_smoke(self):
        master = QuantumMaster(self.units, self.scenarios, [], theta_cap=1024)
        solutions = solve_qubo(master.matrix(), QuantumConfig(initial_temperature=20, alpha=0.9, cutoff_temperature=0.1, iterations_per_temperature=2, candidate_pool_size=4))
        self.assertTrue(solutions)
        self.assertEqual({len(row) for row in solutions}, {master.variable_count()})


class SubproblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
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
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_two_iteration_10x24_run(self):
        config = BendersConfig(
            max_iterations=2,
            quantum=QuantumConfig(initial_temperature=20, alpha=0.9, cutoff_temperature=0.1, iterations_per_temperature=2, candidate_pool_size=4),
        )
        result = solve_benders(self.units, self.scenarios, config)
        self.assertIn(result.status, {"MAX_ITERATIONS", "NO_NEW_CANDIDATES"})
        self.assertTrue(np.isfinite(result.objective))
        self.assertTrue(commitment_is_feasible(self.units, self.scenarios, result.u, result.v, result.w))
        self.assertEqual(len(result.scenarios), 3)
        self.assertGreaterEqual(result.optimality_cuts + result.feasibility_cuts, 1)


if __name__ == "__main__":
    unittest.main()
