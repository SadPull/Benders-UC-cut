from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from suc_benders.candidates import commitment_hash, make_candidate
from suc_benders.data import Scenario
from suc_benders.llm import LLMAdviceResult
from suc_benders.minimal_experiment import configuration
from suc_benders.minimal_report import summarize
from suc_benders.paper_model import key
from suc_benders.rule_advisor import ranked_moves, rule_candidates
from suc_benders.solver import solve_benders
from tests.test_capacity import one_hour_data


class MinimalExperimentTests(unittest.TestCase):
    def setUp(self):
        data = one_hour_data()
        vectors = {k: np.repeat(v, 3) for k, v in vars(data).items()
                   if isinstance(v, np.ndarray) and k != "base_load"}
        self.data = replace(data, **vectors, n_units=3, base_load=np.full(3, 30.))
        self.scenarios = [Scenario("tiny", 1., self.data.base_load)]

    def test_changing_seeds_preserves_base_config_and_keeps_searching(self):
        config = replace(configuration("B", 17, "unused"), max_iterations=4)
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]) as sample:
            result = solve_benders(self.data, self.scenarios, config)
        self.assertEqual([c.args[1].kaiwu_seed for c in sample.call_args_list], [17, 18, 19, 20])
        self.assertEqual(config.quantum.kaiwu_seed, 17)
        self.assertEqual(result.diagnostics["sa_calls"], 4)

    def test_advisor_only_stops_at_three_calls_without_sa(self):
        config = configuration("C", 7, "unused")
        config.llm.api_format = "chat_completions"
        with patch("suc_benders.paper_solver.solve_qubo", side_effect=AssertionError("no SA")), patch(
            "suc_benders.paper_solver.ask_llm_for_candidates", return_value=LLMAdviceResult([], failed=True)) as ask:
            result = solve_benders(self.data, self.scenarios, config)
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(result.status, "LLM_CALL_LIMIT")

    def test_rule_burst_has_same_trigger_and_continues_sa_after_quota(self):
        config = replace(configuration("E", 7, "unused"), max_iterations=7)
        events = []
        config.event_sink = events.append
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
            "suc_benders.paper_solver.ask_llm_for_candidates", side_effect=AssertionError("no LLM")):
            result = solve_benders(self.data, self.scenarios, config)
        self.assertEqual([e["iteration"] for e in events if e["kind"] == "rule_start"], [3, 3, 3])
        self.assertEqual(result.diagnostics["sa_calls"], 7)
        self.assertEqual(result.diagnostics["llm_call_count"], 0)
        self.assertEqual(result.diagnostics["rule_call_count"], 3)
        self.assertTrue(any(e["source"] == "rule" and e["improved"] for e in events if e["kind"] == "evaluation"))

    def test_rule_moves_cover_full_blocks_and_both_trims_and_deduplicate(self):
        incumbent = make_candidate(self.data, self.scenarios, np.ones((3, 3), dtype=int))
        original = incumbent.u.copy()
        context = {"unit_hour_decommit_gradient": []}
        moves = ranked_moves(self.data, incumbent, context)
        intervals = {(left, right) for _, i, left, right in moves if i == 0}
        self.assertEqual(intervals, {(0, 3), (0, 1), (0, 2), (1, 3), (2, 3)})
        result = rule_candidates(self.data, self.scenarios, incumbent, context,
                                 {commitment_hash(incumbent)}, 1, 4, float("inf"))
        self.assertLessEqual(len(result.candidates), 4)
        self.assertEqual(len({commitment_hash(c) for c in result.candidates}), len(result.candidates))
        np.testing.assert_array_equal(incumbent.u, original)

    def test_pure_sa_improvement_is_attributed_to_sa_and_raw_duplicate_is_recorded(self):
        candidate = make_candidate(self.data, self.scenarios, np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))
        point = {key(kind, i, t): int(getattr(candidate, kind)[i, t])
                 for kind in ("u", "v", "w") for i in range(3) for t in range(3)}
        config = replace(configuration("A", 7, "unused"), max_iterations=1)
        events = []
        config.event_sink = events.append
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[[], []]), patch(
            "suc_benders.paper_solver.CompactQubo.decode", return_value=(point, None)):
            result = solve_benders(self.data, self.scenarios, config)
        improved = [e for e in events if e["kind"] == "evaluation" and e["improved"]]
        self.assertEqual(improved[-1]["source"], "sa")
        proposals = [e for e in events if e["kind"] == "proposal"]
        self.assertTrue(proposals[0]["raw_discrete_feasible"])
        self.assertEqual([p["duplicate"] for p in proposals], [False, True])
        self.assertLess(result.objective, improved[0]["objective"])

    def test_report_excludes_late_cost_and_counts_post_llm_sa_gain(self):
        events = [dict(kind="evaluation", elapsed_seconds=t, source=source, objective=cost,
                       feasible=True, improved=True, commitment_hash=str(t))
                  for t, source, cost in [(2, "fallback", 140), (200, "llm", 110),
                                          (500, "sa", 100), (1801, "llm", 90)]]
        report = summarize(events, {}, 1800, 100)
        self.assertEqual(report["objective"], 100)
        self.assertEqual(report["checkpoints"]["300"], 110)
        self.assertEqual(report["threshold_seconds"]["0.005"], 500)
        self.assertEqual(report["sa_improvement_after_llm"], 10)
        self.assertEqual(report["late_evaluations_excluded"], 1)

    def test_unlimited_iterations_really_ends_on_deadline(self):
        config = replace(configuration("A", 7, "unused"), time_limit_seconds=.2)
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]):
            result = solve_benders(self.data, self.scenarios, config)
        self.assertEqual(result.status, "TIME_LIMIT")
        self.assertGreater(result.diagnostics["sa_calls"], 1)


if __name__ == "__main__":
    unittest.main()
