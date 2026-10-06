import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import make_candidate
from suc_benders.context import (
    LLMContextInput,
    CandidateEvaluation,
    build_llm_context,
    _decommit_gradient_rows,
)
from suc_benders.llm import _prompt
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]


class DecommitGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")
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
                candidate_evaluations=[],
                history=[],
            ),
        )
        rows = context["unit_hour_decommit_gradient"]
        expected_on = int(self.incumbent.u.sum())
        # The prompt view is capped; the map must cover every ON cell.
        self.assertLessEqual(len(rows), 60)
        gradient = _decommit_gradient_rows(
            self.units, self.scenarios, self.incumbent, self.results
        )
        self.assertEqual(len(gradient), expected_on)
        for row in gradient:
            self.assertEqual(int(self.incumbent.u[row["unit"], row["period"]]), 1)

    def test_marginal_cost_formula_matches(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                candidate_evaluations=[],
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
                candidate_evaluations=[],
                history=[],
            ),
        )
        raw = {(row["unit"], row["period"]): row for row in _decommit_gradient_rows(
            self.units, self.scenarios, self.incumbent, self.results)}
        for row in context["unit_hour_decommit_gradient"]:
            original = raw[row["unit"], row["period"]]
            expected = -original["fixed_saving"] + original["estimated_fuel_change"]
            self.assertEqual(row["estimated_total_change"], round(expected, 3))

    def test_top_opportunities_capped_at_20(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                candidate_evaluations=[],
                history=[],
            ),
        )
        self.assertLessEqual(len(context["top_decommit_opportunities"]), 20)

    def test_peaker_usage_roles(self):
        context = build_llm_context(
            self.units,
            self.scenarios,
            LLMContextInput(
                incumbent=self.incumbent,
                incumbent_results=self.results,
                base_evaluations=[],
                candidate_evaluations=[],
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
                candidate_evaluations=[],
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
                candidate_evaluations=[
                    CandidateEvaluation("decommit", 590000.0, 110000.0, 480000.0, {}, True, 1, 0, 5)
                ],
                history=[],
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
        ):
            self.assertIn(field, user)
        self.assertIn("removing entire ON blocks", _system)
        self.assertIn("NOT measured", _system)

    def test_cost_decomposition_conserves_scenario_and_shutdown_costs(self):
        from suc_benders.context import _hourly_commitment_profile, _unit_cost_decomposition
        from suc_benders.data import Scenario
        from tests.test_capacity import one_hour_data

        data = one_hour_data()
        scenarios = [Scenario("low", .25, np.array([15., 0., 0.])),
                     Scenario("high", .75, np.array([30., 0., 0.]))]
        incumbent = make_candidate(data, scenarios, np.array([[1, 0, 0]]))
        results = [solve_scenario(data, s, incumbent.u, incumbent.v, incumbent.w)
                   for s in scenarios]
        rows = _unit_cost_decomposition(data, scenarios, incumbent, results)
        expected_fuel = sum(s.probability * r.objective for s, r in zip(scenarios, results))
        self.assertAlmostEqual(sum(row["fuel_cost"] for row in rows), expected_fuel, places=7)
        self.assertEqual(rows[0]["shutdown_cost"], 4.)
        profile = _hourly_commitment_profile(data, incumbent)
        self.assertEqual(profile[1]["units_on"], 0)
        self.assertEqual(profile[1]["hourly_fixed_cost"], 4.)
        fixed = sum(row["fixed_cost"] + row["startup_cost"] + row["shutdown_cost"] for row in rows)
        self.assertEqual(sum(row["hourly_fixed_cost"] for row in profile), fixed)
        context = build_llm_context(data, scenarios, LLMContextInput(
            incumbent, results, [], [], [],
        ))
        self.assertEqual(context["incumbent_cost"]["total"], round(fixed + expected_fuel, 3))


if __name__ == "__main__":
    unittest.main()
