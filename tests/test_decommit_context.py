import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import make_candidate
from suc_benders.context import (
    LocalMoveResult,
    LLMContextInput,
    CandidateEvaluation,
    build_llm_context,
    decommit_gradient_map,
)
from suc_benders.llm import _prompt
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]


class DecommitGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        n, h = cls.units.n_units, cls.units.horizon
        cls.incumbent = make_candidate(
            cls.units, cls.scenarios, np.ones((n, h), dtype=int), source="incumbent"
        )
        cls.incumbent.v[:, 0] = 1 - cls.units.initial_on.astype(int)
        cls.results = [
            solve_scenario(cls.units, sc, cls.incumbent.u, cls.incumbent.v, cls.incumbent.w)
            for sc in cls.scenarios
        ]

    def test_gradient_rows_cover_all_on_cells(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        rows = context["unit_hour_decommit_gradient"]
        expected_on = int(self.incumbent.u.sum())
        # The prompt view is capped; the map must cover every ON cell.
        self.assertLessEqual(len(rows), 60)
        gradient = decommit_gradient_map(
            self.units, self.scenarios, self.incumbent, self.results
        )
        self.assertEqual(len(gradient), expected_on)
        for (i, t) in gradient:
            self.assertEqual(int(self.incumbent.u[i, t]), 1)

    def test_marginal_cost_formula_matches(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        rows = context["unit_hour_decommit_gradient"]
        self.assertTrue(rows)
        for row in rows:
            i = int(row["unit"])
            expected = 2.0 * self.units.cost_a[i] * row["output_mw"] + self.units.cost_b[i]
            # Context values are rounded to 3 decimals by design.
            self.assertAlmostEqual(row["unit_marginal_cost"], expected, places=2)

    def test_total_change_decomposition(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        for row in context["unit_hour_decommit_gradient"]:
            self.assertAlmostEqual(
                row["estimated_total_change"],
                -row["fixed_saving"] + row["estimated_fuel_change"],
                places=3,
            )

    def test_top_opportunities_capped_at_20(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        self.assertLessEqual(len(context["top_decommit_opportunities"]), 20)
        self.assertLessEqual(len(context["decommit_ladder"]), 10)

    def test_peaker_usage_roles(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        roles = context["peaker_usage"]
        self.assertEqual(len(roles), self.units.n_units)
        for row in roles:
            self.assertIn(row["role"], {"base", "mid_merit", "peaker"})
            if row["on_hours"] >= self.units.horizon:
                self.assertEqual(row["role"], "base")

    def test_hourly_profile_exposes_over_commitment(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[],
                candidate_evaluations=[],
                cuts=[],
                history=[],
            ),
        )
        profile = context["hourly_commitment_profile"]
        self.assertEqual(len(profile), self.units.horizon)
        for t, row in enumerate(profile):
            self.assertEqual(row["period"], t)
            self.assertEqual(row["units_on"], int(self.incumbent.u[:, t].sum()))

    def test_prompt_carries_decommit_fields(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                local_move_results=[LocalMoveResult("decommit_block", 9, 20, 4, -500.0, True, "decommit")],
                candidate_evaluations=[
                    CandidateEvaluation("decommit", 590000.0, 110000.0, 480000.0, {}, True, 1, 0, 5)
                ],
                cuts=[],
                history=[],
                decommit_ladder=[
                    CandidateEvaluation("decommit", 590000.0, 110000.0, 480000.0, {}, True, 1, 0, 5)
                ],
            ),
        )
        _system, user = _prompt(
            self.units, self.scenarios, 1, self.incumbent, [], 4, context
        )
        for field in (
            "unit_hour_decommit_gradient",
            "top_decommit_opportunities",
            "hourly_commitment_profile",
            "peaker_usage",
            "decommit_ladder",
        ):
            self.assertIn(field, user)
        self.assertIn("turned OFF", _system)


if __name__ == "__main__":
    unittest.main()
