import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.common import commitment_is_feasible
from suc_benders.exact_master import ExactMasterConfig
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.master import QuantumConfig
from suc_benders.solver import BendersConfig, result_to_dict, solve_benders


ROOT = Path(__file__).resolve().parents[1]


def _fast_quantum():
    return QuantumConfig(
        initial_temperature=20,
        alpha=0.9,
        cutoff_temperature=0.1,
        iterations_per_temperature=2,
        candidate_pool_size=4,
    )


class SolverExactMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        # One shared run each: solving the exact master is the expensive part,
        # so the assertions read from a single pair of Benders runs.
        cls.result = solve_benders(cls.units, cls.scenarios, cls._config())
        cls.legacy = solve_benders(
            cls.units, cls.scenarios, cls._config(exact_master=False)
        )
        cls.exact_full = solve_full(
            cls.units, cls.scenarios, FullConfig(output_flag=0, mip_gap=1e-4)
        )

    @staticmethod
    def _config(**kwargs):
        base = dict(
            max_iterations=2,
            quantum=_fast_quantum(),
            exact_master=True,
            exact_master_config=ExactMasterConfig(time_limit=30.0, output_flag=0),
        )
        base.update(kwargs)
        return BendersConfig(**base)

    def test_history_reports_lower_bound_and_gap(self):
        result = self.result
        self.assertTrue(result.history)
        for record in result.history:
            self.assertTrue(np.isfinite(record.lower_bound))
            self.assertGreaterEqual(record.gap, -1e-8)
            self.assertIn(record.master_status, {"OPTIMAL", "TIME_LIMIT"})

    def test_exact_candidate_is_evaluated(self):
        result = self.result
        self.assertGreater(sum(row.exact_candidates for row in result.history), 0)

    def test_bound_never_exceeds_incumbent(self):
        result = self.result
        self.assertNotEqual(result.status, "BOUND_INCONSISTENCY")
        self.assertLessEqual(result.lower_bound, result.objective + 1e-6)
        self.assertGreaterEqual(result.gap, -1e-8)

    def test_bound_is_below_true_optimum(self):
        self.assertLessEqual(
            self.result.lower_bound, self.exact_full.objective + 1e-6
        )
        # Bounds are only meaningful if they are not trivially weak.
        self.assertGreater(self.result.lower_bound, 0.0)

    def test_incumbent_remains_scenario_feasible(self):
        result = self.result
        self.assertTrue(
            commitment_is_feasible(
                self.units, self.scenarios, result.u, result.v, result.w
            )
        )
        self.assertEqual(len(result.scenarios), 3)
        for solution, scenario in zip(result.scenarios, self.scenarios):
            self.assertEqual(solution.name, scenario.name)
            np.testing.assert_allclose(
                np.sum(solution.p, axis=0), scenario.load, atol=1e-5
            )
            self.assertGreaterEqual(
                float(np.sum(solution.r, axis=0).min()),
                self.units.reserve_fraction * float(scenario.load.min()) - 1e-5,
            )

    def test_disabling_exact_master_restores_old_path(self):
        result = self.legacy
        self.assertEqual(result.master_status, "DISABLED")
        self.assertEqual(result.master_runtime, 0.0)
        self.assertFalse(np.isfinite(result.lower_bound))
        self.assertEqual(sum(row.exact_candidates for row in result.history), 0)

    def test_result_serializes_new_fields(self):
        result = self.result
        payload = result_to_dict(result)
        for key in ("lower_bound", "gap", "master_status", "master_runtime"):
            self.assertIn(key, payload)
        for record in payload["history"]:
            for key in ("lower_bound", "gap", "exact_candidates", "master_status"):
                self.assertIn(key, record)


if __name__ == "__main__":
    unittest.main()
