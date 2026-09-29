import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import decommit_commitment
from suc_benders.common import commitment_is_feasible
from suc_benders.llm import LLMAdviceResult, LLMConfig
from suc_benders.master import QuantumConfig
from suc_benders.solver import (
    BendersConfig,
    _construct_feasible_candidate,
    result_to_dict,
    solve_benders,
)
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]
REMOVED_FIELDS = {
    "lower_bound", "gap", "master_status", "master_runtime",
    "exact_candidates", "perturbed_candidates",
}


class SolverIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def setUp(self):
        self.optimized_models = []
        original_optimize = gp.Model.optimize

        def continuous_only(model, *args, **kwargs):
            model.update()
            self.assertTrue(all(var.VType == gp.GRB.CONTINUOUS for var in model.getVars()))
            self.optimized_models.append(model.ModelName)
            return original_optimize(model, *args, **kwargs)

        guard = patch.object(gp.Model, "optimize", continuous_only)
        guard.start()
        self.addCleanup(guard.stop)
        reference_guard = patch(
            "suc_benders.full_model.solve_full", side_effect=AssertionError("reference called")
        )
        reference_guard.start()
        self.addCleanup(reference_guard.stop)

    def test_default_generators_only_optimize_continuous_subproblems(self):
        captured = []

        def advice(*args, **kwargs):
            captured.append(kwargs["context"])
            return LLMAdviceResult([], 0, 0, False, "ok")

        config = BendersConfig(
            max_iterations=1,
            quantum=QuantumConfig(
                initial_temperature=20, alpha=0.9, cutoff_temperature=0.1,
                iterations_per_temperature=2, candidate_pool_size=4,
            ),
            llm=LLMConfig(enabled=True),
        )
        with patch("suc_benders.solver.ask_llm_for_candidates", side_effect=advice):
            result = solve_benders(self.units, self.scenarios, config)
        self.assertTrue(self.optimized_models)
        self.assertTrue(np.isfinite(result.objective))
        self.assertEqual(result.status, "MAX_ITERATIONS")
        self.assertGreater(result.history[0].decommit_candidates, 0)
        self.assertGreater(result.optimality_cuts, 0)
        self.assertTrue(commitment_is_feasible(
            self.units, self.scenarios, result.u, result.v, result.w,
        ))
        for solution, scenario in zip(result.scenarios, self.scenarios):
            np.testing.assert_allclose(solution.p.sum(axis=0), scenario.load, atol=1e-5)
            self.assertTrue(np.all(
                solution.r.sum(axis=0) >= self.units.reserve_fraction * scenario.load - 1e-5
            ))
        allowed_sources = {"quantum", "repair", "fallback", "incumbent", "decommit", "llm"}
        for key in ("base_candidates", "candidate_comparison", "decommit_ladder"):
            self.assertTrue(all(row["source"] in allowed_sources for row in captured[0][key]))
        self.assertTrue(captured[0]["decommit_ladder"])
        payload = result_to_dict(result)
        self.assertFalse(REMOVED_FIELDS.intersection(payload))
        for record in payload["history"]:
            self.assertFalse(REMOVED_FIELDS.intersection(record))
        json.dumps(payload, allow_nan=False)

    def test_no_annealing_deduplicates_fallback_and_incumbent(self):
        with patch("suc_benders.solver.QuantumMaster", side_effect=AssertionError("QUBO called")), \
             patch("suc_benders.solver.solve_qubo", side_effect=AssertionError("SA called")), \
             patch("suc_benders.solver.ask_llm_for_candidates", side_effect=AssertionError("LLM called")), \
             patch("suc_benders.solver.solve_scenario", wraps=solve_scenario) as scenario_solver:
            result = solve_benders(self.units, self.scenarios, BendersConfig(
                max_iterations=5, quantum_candidates=False, decommit_candidates=False,
            ))
        self.assertEqual(result.status, "NO_NEW_CANDIDATES")
        self.assertEqual([row.new_candidates for row in result.history], [1, 0])
        self.assertEqual(scenario_solver.call_count, len(self.scenarios))
        self.assertEqual(result.optimality_cuts, 1)

    def test_llm_failure_retries_before_stopping_on_duplicates(self):
        failed = LLMAdviceResult([], 0, 0, True, "temporary failure")
        success = LLMAdviceResult([], 0, 0, False, "ok")
        with patch("suc_benders.solver.ask_llm_for_candidates", side_effect=[failed, failed, success]) as ask:
            result = solve_benders(self.units, self.scenarios, BendersConfig(
                max_iterations=4, quantum_candidates=False, decommit_candidates=False,
                llm=LLMConfig(enabled=True),
            ))
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(result.status, "NO_NEW_CANDIDATES")
        self.assertEqual([row.llm_failed for row in result.history], [True, True, False])
        self.assertEqual([row.new_candidates for row in result.history], [1, 0, 0])

    def test_llm_candidate_is_evaluated_and_deduplicated(self):
        base = _construct_feasible_candidate(self.units, self.scenarios)
        proposal = decommit_commitment(self.units, self.scenarios, base, limit=1)[0]
        proposal.source = "llm"
        with patch("suc_benders.solver.ask_llm_for_candidates", return_value=LLMAdviceResult(
            [proposal, proposal], 2, 2, False, "ok",
        )), patch("suc_benders.solver.solve_scenario", wraps=solve_scenario) as scenario_solver:
            result = solve_benders(self.units, self.scenarios, BendersConfig(
                max_iterations=1, quantum_candidates=False, decommit_candidates=False,
                llm=LLMConfig(enabled=True),
            ))
        self.assertEqual(result.history[0].new_candidates, 2)
        self.assertEqual(scenario_solver.call_count, 2 * len(self.scenarios))
        self.assertEqual(result.optimality_cuts, 2)

    def test_no_candidate_does_not_claim_model_infeasibility(self):
        with patch("suc_benders.solver._construct_feasible_candidate", return_value=None):
            result = solve_benders(self.units, self.scenarios, BendersConfig(
                max_iterations=3, quantum_candidates=False,
            ))
        self.assertEqual(result.status, "NO_FEASIBLE_SOLUTION")
        self.assertFalse(self.optimized_models)
        self.assertEqual(result.u.size, 0)
        self.assertIsNone(result_to_dict(result)["objective"])

    def test_failed_scenarios_return_continuous_feasibility_cuts(self):
        units = replace(self.units, ramp_up=np.zeros(self.units.n_units))
        result = solve_benders(units, self.scenarios, BendersConfig(
            max_iterations=1, quantum_candidates=False,
        ))
        self.assertEqual(result.status, "NO_FEASIBLE_SOLUTION")
        self.assertEqual(result.feasibility_cuts, len(self.scenarios))
        self.assertGreater(len(self.optimized_models), len(self.scenarios))
        self.assertEqual(result.optimality_cuts, 0)


if __name__ == "__main__":
    unittest.main()
