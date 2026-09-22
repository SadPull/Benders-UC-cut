import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from suc_benders import load_suc_data
from suc_benders.llm import LLMAdviceResult, LLMConfig
from suc_benders.master import QuantumConfig
from suc_benders.solver import BendersConfig, solve_benders


ROOT = Path(__file__).resolve().parents[1]


def _fast_quantum():
    return QuantumConfig(
        initial_temperature=20,
        alpha=0.9,
        cutoff_temperature=0.1,
        iterations_per_temperature=2,
        candidate_pool_size=4,
    )


class SolverDecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_decommit_runs_when_enabled(self):
        result = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=2,
                quantum=_fast_quantum(),
                decommit_candidates=True,
                perturb_candidates=False,
            ),
        )
        total = sum(row.decommit_candidates for row in result.history)
        self.assertGreater(total, 0)
        self.assertTrue(np.isfinite(result.objective))

    def test_decommit_disabled_produces_no_decommit_candidates(self):
        result = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=2,
                quantum=_fast_quantum(),
                decommit_candidates=False,
                perturb_candidates=False,
            ),
        )
        total = sum(row.decommit_candidates for row in result.history)
        self.assertEqual(total, 0)

    def test_decommit_disabled_still_no_perturbation(self):
        result = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=2,
                quantum=_fast_quantum(),
                decommit_candidates=False,
                perturb_candidates=False,
            ),
        )
        self.assertEqual(sum(row.perturbed_candidates for row in result.history), 0)

    def test_ladder_is_passed_to_llm_context(self):
        captured = {}

        def fake_ask(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return LLMAdviceResult([], 0, 0, False, "ok")

        with patch("suc_benders.solver.ask_llm_for_candidates", side_effect=fake_ask):
            solve_benders(
                self.units,
                self.scenarios,
                BendersConfig(
                    max_iterations=1,
                    quantum=_fast_quantum(),
                    decommit_candidates=True,
                    perturb_candidates=False,
                    llm=LLMConfig(enabled=True),
                ),
            )
        context = captured["context"]
        self.assertIn("decommit_ladder", context)
        self.assertTrue(context["decommit_ladder"])
        self.assertIn("unit_hour_decommit_gradient", context)
        self.assertIn("top_decommit_opportunities", context)
        self.assertIn("peaker_usage", context)

    def test_decommit_shortens_horizon_on_hours(self):
        """Decommitment must not increase total on-hours vs the repair baseline."""
        baseline = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=2,
                quantum=_fast_quantum(),
                decommit_candidates=False,
                perturb_candidates=False,
            ),
        )
        decommitted = solve_benders(
            self.units,
            self.scenarios,
            BendersConfig(
                max_iterations=2,
                quantum=_fast_quantum(),
                decommit_candidates=True,
                perturb_candidates=False,
            ),
        )
        self.assertLessEqual(
            int(decommitted.u.sum()), int(baseline.u.sum())
        )


if __name__ == "__main__":
    unittest.main()
