import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from suc_benders import Scenario, load_suc_data
from suc_benders.common import commitment_is_feasible
from suc_benders.exact_master import ExactMasterConfig, solve_exact_master
from suc_benders.candidates import make_candidate, repair_commitment
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

        points = [make_candidate(self.units, self.scenarios, u)]
        rng = np.random.default_rng(0)
        for _ in range(4):
            attempt = u.copy()
            mask = rng.random((n, horizon)) < 0.35
            attempt[mask] = 0
            repaired = repair_commitment(self.units, self.scenarios, attempt)
            if repaired is not None:
                points.append(repaired)
        self.assertGreater(len(points), 1)

        checked = 0
        for point in points:
            self.assertTrue(commitment_is_feasible(
                self.units, self.scenarios, point.u, point.v, point.w
            ))
            results = [
                solve_scenario(self.units, scenario, point.u, point.v, point.w)
                for scenario in self.scenarios
            ]
            self.assertTrue(all(result.feasible for result in results))
            expected = sum(
                scenario.probability * result.objective
                for scenario, result in zip(self.scenarios, results)
            )
            self.assertLessEqual(cut_value(aggregated, point.u, point.v, point.w), expected + 1e-6)
            checked += 1
        self.assertEqual(checked, len(points))
        self.assertTrue(any(np.any(point.v) or np.any(point.w) for point in points))

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

    def test_loose_optimal_status_reports_objbound(self):
        u = np.ones((self.units.n_units, self.units.horizon), dtype=int)
        zeros = np.zeros_like(u)
        cut, _ = self._aggregated_cut(self.units, self.scenarios, u, zeros, zeros)
        result = solve_exact_master(
            self.units, self.scenarios, [cut],
            ExactMasterConfig(time_limit=30.0, mip_gap=0.1),
        )
        self.assertEqual(result.status, "OPTIMAL")
        self.assertAlmostEqual(result.bound, result.model.ObjBound, places=6)
        exact = solve_full(self.units, self.scenarios, FullConfig(output_flag=0))
        self.assertLessEqual(result.bound, exact.objective + 1e-6)

    def test_zero_time_limit_without_master_incumbent(self):
        result = solve_exact_master(
            self.units, self.scenarios, [], ExactMasterConfig(time_limit=0.0)
        )
        self.assertEqual(result.status, "TIME_LIMIT")
        self.assertIsNone(result.u)
        self.assertEqual(result.objective, float("inf"))
        self.assertEqual(result.mip_gap, float("inf"))

    def test_initial_ramp_cover_preserves_small_instance_feasible_points(self):
        # Unit 0 alone has enough nameplate capacity but cannot supply hour 0.
        fields = {
            key: value[:2].copy() for key, value in vars(self.units).items()
            if isinstance(value, np.ndarray)
        }
        fields.update(
            pmax=np.array([100., 80.]), pmin=np.array([10., 10.]),
            pstart=np.array([100., 80.]), pshut=np.array([100., 80.]),
            ramp_up=np.array([30., 50.]), ramp_down=np.array([80., 80.]),
            min_up=np.ones(2, dtype=int), min_down=np.ones(2, dtype=int),
            initial_on=np.ones(2, dtype=bool), initial_duration=np.ones(2),
            initial_output=np.array([10., 10.]), base_load=np.array([60., 70.]),
        )
        units = replace(self.units, n_units=2, horizon=2, **fields)
        scenarios = [Scenario("small", 1.0, units.base_load)]
        master = solve_exact_master(units, scenarios, [], self.config)
        row = master.model.getConstrByName("initial_ramp_cover")
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.RHS, 60.0)  # No reserve factor here.
        for i, coefficient in enumerate([40.0, 60.0]):
            self.assertAlmostEqual(
                master.model.getCoeff(row, master.model.getVarByName(f"u[{i},0]")),
                coefficient,
            )
        feasible_count = 0
        rejected_count = 0
        for bits in product([0, 1], repeat=4):
            candidate = make_candidate(units, scenarios, np.array(bits).reshape(2, 2))
            if not commitment_is_feasible(units, scenarios, candidate.u, candidate.v, candidate.w):
                continue
            sub = solve_scenario(units, scenarios[0], candidate.u, candidate.v, candidate.w)
            if not sub.feasible:
                self.assertGreater(cut_value(sub, candidate.u, candidate.v, candidate.w), 0.0)
                self.assertLessEqual(cut_value(sub, np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))), 1e-6)
                rejected_count += 1
                continue
            feasible_count += 1
            for name in ("u", "v", "w"):
                for i in range(2):
                    for t in range(2):
                        var = master.model.getVarByName(f"{name}[{i},{t}]")
                        var.LB = var.UB = int(getattr(candidate, name)[i, t])
            master.model.optimize()
            self.assertEqual(master.model.Status, 2)
        self.assertGreater(feasible_count, 1)
        self.assertGreater(rejected_count, 0)


if __name__ == "__main__":
    unittest.main()
