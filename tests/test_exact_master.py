import unittest
from pathlib import Path

import numpy as np

from suc_benders import Scenario, load_suc_data
from suc_benders.common import commitment_is_feasible
from suc_benders.exact_master import ExactMasterConfig, solve_exact_master
from suc_benders.candidates import repair_commitment
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.master import BendersCut
from suc_benders.subproblem import (
    aggregate_optimality_cut,
    cut_value,
    solve_scenario,
)


ROOT = Path(__file__).resolve().parents[1]


class ExactMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        cls.config = ExactMasterConfig(time_limit=60.0, output_flag=0)

    def test_master_variable_count_and_theta_continuous(self):
        result = solve_exact_master(self.units, self.scenarios, [], self.config)
        n, horizon = self.units.n_units, self.units.horizon
        binaries = [
            var for var in result.model.getVars() if var.VType == "B"
        ]
        self.assertEqual(len(binaries), 3 * n * horizon)
        self.assertEqual(len(result.model.getVars()), 3 * n * horizon + 1)
        theta_var = result.model.getVarByName("theta")
        self.assertEqual(theta_var.VType, "C")
        self.assertAlmostEqual(theta_var.LB, 0.0)

    def test_master_candidate_is_discrete_feasible(self):
        result = solve_exact_master(self.units, self.scenarios, [], self.config)
        self.assertIsNotNone(result.u)
        self.assertTrue(
            commitment_is_feasible(self.units, self.scenarios, result.u, result.v, result.w)
        )
        self.assertAlmostEqual(result.theta, 0.0, places=6)

    @staticmethod
    def _aggregated_cut(units, scenarios, u, v, w):
        results = [solve_scenario(units, scenario, u, v, w) for scenario in scenarios]
        for result in results:
            assert result.feasible
        aggregated = aggregate_optimality_cut(results, scenarios)
        return BendersCut(
            aggregated.cut_alpha,
            aggregated.cut_beta_u,
            aggregated.cut_beta_v,
            aggregated.cut_beta_w,
            "optimality",
        ), aggregated

    def test_optimality_cut_constrains_theta(self):
        n, horizon = self.units.n_units, self.units.horizon
        u = np.ones((n, horizon), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        cut, aggregated = self._aggregated_cut(self.units, self.scenarios, u, v, w)
        result = solve_exact_master(self.units, self.scenarios, [cut], self.config)
        self.assertAlmostEqual(
            float(cut_value(aggregated, u, v, w)), aggregated.objective, places=4
        )
        # theta must equal the cut value at the master optimum: the aggregated
        # cut is the only lower bound on the expected second-stage cost.
        at_optimum = float(cut_value(aggregated, result.u, result.v, result.w))
        self.assertAlmostEqual(result.theta, at_optimum, places=3)
        self.assertGreaterEqual(result.bound, 0.0)

    def test_aggregated_cut_underestimates_expected_fuel(self):
        """A Benders cut must never exceed the true expected subproblem cost."""
        n, horizon = self.units.n_units, self.units.horizon
        u = np.ones((n, horizon), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        cut, aggregated = self._aggregated_cut(self.units, self.scenarios, u, v, w)

        points = [u]
        rng = np.random.default_rng(0)
        for _ in range(4):
            attempt = u.copy()
            mask = rng.random((n, horizon)) < 0.35
            attempt[mask] = 0
            repaired = repair_commitment(self.units, self.scenarios, attempt)
            if repaired is not None:
                points.append(repaired.u)
        self.assertGreater(len(points), 1)

        checked = 0
        for point in points:
            if not commitment_is_feasible(self.units, self.scenarios, point, v, w):
                continue
            results = [
                solve_scenario(self.units, scenario, point, v, w)
                for scenario in self.scenarios
            ]
            self.assertTrue(all(result.feasible for result in results))
            expected = sum(
                scenario.probability * result.objective
                for scenario, result in zip(self.scenarios, results)
            )
            self.assertLessEqual(cut_value(aggregated, point, v, w), expected + 1e-6)
            checked += 1
        self.assertGreater(checked, 0)

    def test_adding_cut_does_not_decrease_bound(self):
        base = solve_exact_master(self.units, self.scenarios, [], self.config)
        n, horizon = self.units.n_units, self.units.horizon
        u = np.ones((n, horizon), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        cut, _ = self._aggregated_cut(self.units, self.scenarios, u, v, w)
        tightened = solve_exact_master(self.units, self.scenarios, [cut], self.config)
        self.assertGreaterEqual(tightened.bound, base.bound - 1e-6)

    def test_bound_is_below_true_optimum(self):
        """The whole point of the exact master: a bound that is never optimistic."""
        n, horizon = self.units.n_units, self.units.horizon
        u = np.ones((n, horizon), dtype=int)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        cut, _ = self._aggregated_cut(self.units, self.scenarios, u, v, w)
        result = solve_exact_master(self.units, self.scenarios, [cut], self.config)
        exact = solve_full(self.units, self.scenarios, FullConfig(output_flag=0, mip_gap=1e-6))
        self.assertLessEqual(result.bound, exact.objective + 1e-6)

    def test_feasibility_cut_excludes_infeasible_commitment(self):
        n, horizon = self.units.n_units, self.units.horizon
        impossible = Scenario(
            "impossible", 1.0, np.full(horizon, self.units.pmax.sum() + 1.0)
        )
        scenarios = [*self.scenarios, impossible]
        all_on = np.ones((n, horizon), dtype=int)
        zeros = np.zeros_like(all_on)
        sub = solve_scenario(self.units, impossible, all_on, zeros, zeros)
        self.assertFalse(sub.feasible)
        cut = BendersCut(
            sub.cut_alpha, sub.cut_beta_u, sub.cut_beta_v, sub.cut_beta_w, "feasibility"
        )
        value = float(
            cut.alpha + np.sum(cut.beta_u * all_on) + np.sum(cut.beta_v * zeros) + np.sum(cut.beta_w * zeros)
        )
        self.assertGreater(value, 0.0)

    def test_time_limit_reports_bound_not_objective(self):
        config = ExactMasterConfig(time_limit=0.01, mip_gap=1e-9, output_flag=0)
        result = solve_exact_master(self.units, self.scenarios, [], config)
        self.assertIn(result.status, {"TIME_LIMIT", "OPTIMAL"})
        if result.status == "TIME_LIMIT":
            self.assertLessEqual(result.bound, result.objective + 1e-6)


if __name__ == "__main__":
    unittest.main()
