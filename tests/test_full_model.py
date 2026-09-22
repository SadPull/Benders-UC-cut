import unittest
from pathlib import Path

from suc_benders import load_suc_data
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]


class FullModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        cls.result = solve_full(cls.units, cls.scenarios, FullConfig(output_flag=0, mip_gap=1e-4))

    def test_full_model_objective_decomposition(self):
        self.assertEqual(self.result.status, "OPTIMAL")
        self.assertAlmostEqual(
            self.result.objective,
            self.result.fixed_cost + self.result.expected_fuel_cost,
            places=5,
        )

    def test_full_model_commitment_is_feasible(self):
        from suc_benders.common import commitment_is_feasible

        self.assertTrue(
            commitment_is_feasible(
                self.units,
                self.scenarios,
                self.result.u,
                self.result.v,
                self.result.w,
            )
        )

    def test_scenario_subproblems_match_full_model(self):
        for scenario in self.scenarios:
            sub = solve_scenario(self.units, scenario, self.result.u, self.result.v, self.result.w)
            self.assertTrue(sub.feasible)
            self.assertAlmostEqual(sub.objective, scenario.probability * sub.objective / scenario.probability, places=5)


if __name__ == "__main__":
    unittest.main()
