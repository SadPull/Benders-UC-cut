import unittest
from pathlib import Path

import numpy as np

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

    def test_reference_bound_uses_objbound_even_with_nonzero_mip_gap(self):
        self.assertTrue(np.isfinite(self.result.lower_bound))
        self.assertEqual(self.result.lower_bound, self.result.model.ObjBound)
        self.assertLessEqual(self.result.lower_bound, self.result.objective + 1e-6)

    def test_reference_timeout_without_incumbent(self):
        result = solve_full(self.units, self.scenarios, FullConfig(time_limit=0, output_flag=0))
        self.assertEqual(result.status, "TIME_LIMIT")
        self.assertEqual(result.objective, float("inf"))
        self.assertEqual(result.scenarios, [])
        self.assertEqual(result.mip_gap, float("inf"))
        self.assertFalse(np.isnan(result.lower_bound))

    def test_scenario_subproblems_match_full_model(self):
        for scenario in self.scenarios:
            sub = solve_scenario(self.units, scenario, self.result.u, self.result.v, self.result.w)
            self.assertTrue(sub.feasible)
            self.assertAlmostEqual(sub.objective, scenario.probability * sub.objective / scenario.probability, places=5)


if __name__ == "__main__":
    unittest.main()
