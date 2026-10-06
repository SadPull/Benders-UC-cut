import math
import time
import unittest
from unittest.mock import patch

from suc_benders.pure_sa_gap_experiment import PHASE_SECONDS, TOTAL_SECONDS, configuration, definitions
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.validation import validate_schedule
from tests.test_temporal_solver import enumerated_run, tiny


class PureSAIsolationTests(unittest.TestCase):
    def test_pure_search_never_calls_integer_oracles(self):
        data, scenarios = tiny(1)
        emitted = []
        with patch("suc_benders.paper_solver.solve_master", side_effect=AssertionError("integer candidate")), patch(
                "suc_benders.paper_solver.certify_master", side_effect=AssertionError("integer certificate")):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, max_inner_iterations=2,
                gap_experiment_variant="sa_strict", defer_full_recovery=True, event_sink=emitted.append))
        validate_schedule(data, scenarios, result)
        self.assertEqual(result.diagnostics["online_integer_master_calls"], 0)
        self.assertTrue(any(r["kind"] == "multiplier_epoch" for r in emitted))
        self.assertTrue(any(r["kind"] == "local_cut" for r in emitted))
        self.assertLessEqual(result.diagnostics["reliable_lower"], result.objective+1e-5)

    def test_pure_variant_rejects_native_candidate_configuration(self):
        data, scenarios = tiny(1)
        for variant in ("sa_strict", "sa_flow", "sa_compact", "sa_full"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                solve_benders(data, scenarios, BendersConfig(gap_experiment_variant=variant,
                              native_fallback_patience=3))
        with self.assertRaises(ValueError):
            solve_benders(data, scenarios, BendersConfig(gap_experiment_variant="unknown"))

    def test_fixed_matrix_and_budget(self):
        self.assertEqual(TOTAL_SECONDS, 14400)
        runs = definitions()
        self.assertEqual(len(runs), 23)
        for stage, seconds in PHASE_SECONDS.items():
            if stage not in ("reference", "audit"):
                self.assertEqual(sum(r["seconds"]+r["audit_seconds"] for r in runs if r["stage"] == stage), seconds)
        for run in runs:
            if run["method"] == "selected":
                with self.assertRaises(ValueError):
                    configuration(run)
                continue
            cfg = configuration({**run, "hours": run["hours"] or 1})
            native = run["method"] == "native_diagnostic"
            self.assertEqual(cfg.quantum_candidates, not native)
            self.assertEqual(cfg.native_fallback_patience, 1 if native else None)
            self.assertEqual(cfg.quantum.candidate_pool_size, 16)
            self.assertEqual(cfg.max_inner_iterations, 200 if run["method"] == "sa_strict" else 8)
            self.assertFalse(cfg.llm.enabled)
        final = [r for r in runs if r["stage"] == "confirmation"]
        self.assertEqual([r["seed"] for r in final], [7, 100007, 200007])
        self.assertTrue(all(r["seconds"] == 1200 and r["audit_seconds"] == 600 for r in final))


if __name__ == "__main__":
    unittest.main()
