import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from suc_benders import load_suc_data, Scenario
from suc_benders.candidates import make_candidate
from suc_benders.context import build_llm_context, CandidateEvaluation, LLMContextInput
from suc_benders.llm import LLMAdviceResult, LLMConfig, _prompt
from suc_benders.master import QuantumConfig
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]


class ScenarioDualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")
        n, h = cls.units.n_units, cls.units.horizon
        cls.u = np.ones((n, h), dtype=int)
        cls.v = np.zeros_like(cls.u)
        cls.w = np.zeros_like(cls.u)
        cls.v[:, 0] = 1 - cls.units.initial_on.astype(int)

    def test_feasible_scenario_returns_duals(self):
        result = solve_scenario(self.units, self.scenarios[1], self.u, self.v, self.w)
        self.assertTrue(result.feasible)
        self.assertIsNotNone(result.balance_dual)
        self.assertIsNotNone(result.reserve_dual)
        self.assertEqual(result.balance_dual.shape, (24,))
        self.assertEqual(result.reserve_dual.shape, (24,))
        self.assertTrue(np.all(np.isfinite(result.balance_dual)))
        self.assertTrue(np.all(np.isfinite(result.reserve_dual)))

    def test_infeasible_scenario_keeps_duals_none(self):
        impossible = Scenario("impossible", 1.0, np.full(24, self.units.pmax.sum() + 1.0))
        result = solve_scenario(self.units, impossible, self.u, self.v, self.w)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.balance_dual)
        self.assertIsNone(result.reserve_dual)


class PromptContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

    def test_prompt_contains_context(self):
        context = {
            "incumbent_cost": {"total": 1.0},
            "base_candidates": [],
            "scenario_duals": {"balance": [[0.0]], "reserve": [[0.0]]},
        }
        _system, user = _prompt(
            self.units,
            self.scenarios,
            1,
            None,
            [],
            4,
            context,
        )
        body = json.loads(user)
        self.assertIn("context", body)
        self.assertEqual(body["context"], context)
        self.assertIn("candidates", body["output_schema"])


class SolverContextIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

    def test_solver_passes_p0_p1_context_to_llm(self):
        captured = {}

        def fake_ask_llm(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return LLMAdviceResult([], 0, 0, False, "ok")

        with patch("suc_benders.paper_solver.ask_llm_for_candidates", side_effect=fake_ask_llm), \
             patch("suc_benders.paper_solver.solve_qubo", return_value=[]):
            result = solve_benders(
                self.units,
                self.scenarios,
                BendersConfig(
                    max_iterations=1,
                    quantum=QuantumConfig(
                        initial_temperature=20,
                        alpha=0.9,
                        cutoff_temperature=0.1,
                        iterations_per_temperature=2,
                        candidate_pool_size=4,
                    ),
                    llm=LLMConfig(enabled=True),
                ),
            )
        self.assertTrue(np.isfinite(result.objective))
        self.assertIn("context", captured)
        context = captured["context"]
        for key in (
            "incumbent_cost",
            "base_candidates",
            "scenario_duals",
            "candidate_comparison",
            "scenario_dispatch_summary",
            "reserve_margin",
            "tight_constraints",
            "infeasible_diagnostics",
            "iteration_history",
            "unit_marginal_cost",
            "unit_cost_decomposition",
        ):
            self.assertIn(key, context)
        self.assertEqual(len(context["scenario_duals"]["balance"]), 3)
        self.assertEqual(len(context["scenario_duals"]["reserve"]), 3)
        self.assertEqual(len(context["reserve_margin"]), 3)
        self.assertLessEqual(len(context["base_candidates"]), 10)
        self.assertLessEqual(len(context["candidate_comparison"]), 10)
        self.assertLessEqual(len(context["tight_constraints"]), 20)
        self.assertLessEqual(len(context["infeasible_diagnostics"]), 20)
        self.assertLessEqual(len(context["iteration_history"]), 5)

    def test_llm_disabled_path_remains_local(self):
        result = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=1,
                quantum=QuantumConfig(
                    initial_temperature=20,
                    alpha=0.9,
                    cutoff_temperature=0.1,
                    iterations_per_temperature=2,
                    candidate_pool_size=4,
                ),
            ),
        )
        self.assertTrue(np.isfinite(result.objective))
        self.assertGreaterEqual(result.optimality_cuts, 1)


class ContextBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

    def test_context_values_are_rounded_and_finite(self):
        incumbent = make_candidate(
            self.units,
            self.scenarios,
            np.ones((self.units.n_units, self.units.horizon), dtype=int),
            source="incumbent",
        )
        incumbent.v[:, 0] = 1 - self.units.initial_on.astype(int)
        results = [
            solve_scenario(self.units, scenario, incumbent.u, incumbent.v, incumbent.w)
            for scenario in self.scenarios
        ]
        evaluation = CandidateEvaluation(
            source="incumbent",
            total_cost=123.456789,
            fixed_cost=23.456789,
            expected_fuel_cost=100.0,
            scenario_costs={scenario.name: 100.0 for scenario in self.scenarios},
            feasible=True,
            optimality_cuts=1,
            feasibility_cuts=0,
            hamming_to_incumbent=0,
        )
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=incumbent,
                incumbent_results=results,
                base_evaluations=[evaluation],
                candidate_evaluations=[evaluation],
                history=[],
                infeasible_diagnostics=[],
            ),
        )
        fixed = float(
            np.sum(self.units.cost_c[:, None] * incumbent.u)
            + np.sum(self.units.startup_cost[:, None] * incumbent.v)
            + np.sum(self.units.shutdown_cost[:, None] * incumbent.w)
        )
        expected_fuel = sum(
            scenario.probability * result.objective
            for scenario, result in zip(self.scenarios, results)
        )
        self.assertEqual(context["incumbent_cost"]["total"], round(fixed + expected_fuel, 3))
        self.assertEqual(context["base_candidates"][0]["total_cost"], 123.457)
        self.assertEqual(len(context["unit_marginal_cost"]), self.units.n_units)
        self.assertEqual(len(context["unit_cost_decomposition"]), self.units.n_units)


if __name__ == "__main__":
    unittest.main()
