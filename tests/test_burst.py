import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from suc_benders.advisor_feedback import call_due
from suc_benders.attribution import validate_schedule
from suc_benders.candidates import make_candidate
from suc_benders.data import Scenario
from suc_benders.llm import LLMAdviceResult, LLMConfig
from suc_benders.solver import BendersConfig, _fixed_cost, solve_benders
from suc_benders.subagent import initialize, publish, receipt
from suc_benders.subproblem import solve_scenario
from tests.test_capacity import one_hour_data


class BurstTests(unittest.TestCase):
    def setUp(self):
        original = one_hour_data()
        vectors = {k: np.repeat(v, 3) for k, v in vars(original).items()
                   if isinstance(v, np.ndarray) and k != "base_load"}
        vectors["cost_c"] = np.full(3, 50.)
        self.data = replace(original, **vectors, n_units=3, base_load=np.full(3, 30.))
        self.scenarios = [Scenario("tiny", 1., self.data.base_load)]
        self.candidates = [make_candidate(self.data, self.scenarios,
            np.array([[1, 1, 1]] * n + [[0, 0, 0]] * (3-n)), source="llm") for n in (3, 2, 1)]
        self.config = BendersConfig(master_encoding="dynamic", decommit_candidates=False,
            max_iterations=6, stop_on_no_new_candidates=False,
            llm=LLMConfig(enabled=True, schedule="stagnation_burst", enhanced_feedback=True))

    def test_first_burst_requires_three_consecutive_insufficient_rounds(self):
        stalled = 0
        for expected in (False, False, True):
            due, stalled = call_due("stagnation_burst", 2000., 2000., stalled, 0)
            self.assertEqual(due, expected)
        # Exactly 0.05% improvement is sufficient and clears stagnation.
        self.assertEqual(call_due("stagnation_burst", 2000., 1999., stalled, 0), (False, 0))

    def test_failed_burst_returns_to_three_sa_rounds_and_respects_call_limit(self):
        events = []
        self.config.event_sink = events.append
        self.config.max_iterations = 9
        self.config.llm.max_calls = 4
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates",
                return_value=LLMAdviceResult([], failed=True)) as ask:
            result = solve_benders(self.data, self.scenarios, self.config)
        self.assertEqual([call.args[2] for call in ask.call_args_list], [1, 2, 3, 4])
        self.assertEqual([e["iteration"] for e in events if e["kind"] == "llm_start"], [3, 3, 3, 6])
        self.assertEqual([e["stalled"] for e in events if e["kind"] == "sa_round_validated"],
                         [1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertEqual(result.diagnostics["sa_calls"], 9)

    def test_each_burst_input_uses_validated_best_and_previous_feedback(self):
        events = []
        self.config.event_sink = events.append
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            initialize(root, {})
            self.config.llm = replace(self.config.llm, api_format="codex_subagent",
                exchange_dir=str(root), timeout_seconds=None)
            def respond(_):
                for call in sorted((root / "calls").iterdir()):
                    if (call / "receipt.json").exists():
                        continue
                    index = int(call.name)
                    candidate = self.candidates[min(index, 2)]
                    publish(call / "candidates.json", {"candidates": [{"u": candidate.u.tolist()}]})
                    receipt(root, index, f"test-agent-{index}", "completed")
            with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                    "suc_benders.subagent.time.sleep", side_effect=respond):
                result = solve_benders(self.data, self.scenarios, self.config)
            for index in range(2, 7):
                user = json.loads((root / f"calls/{index:03d}/user.txt").read_text())
                feedback = json.loads((root / f"calls/{index-1:03d}/feedback.json").read_text())
                self.assertEqual(user["context"]["previous_call_feedback"], feedback)
                self.assertEqual(user["incumbent_u"], self.candidates[min(index-1, 2)].u.tolist())
            self.assertEqual(result.diagnostics["llm_call_count"], 6)
            self.assertEqual(result.diagnostics["sa_calls"], 6)
            self.assertEqual([e["kind"] for e in events if e["kind"] in {"sa_start", "llm_start"}],
                             (["sa_start"] * 3 + ["llm_start"] * 3) * 2)

    def test_target_stops_inside_any_call_without_finishing_batch_or_candidates(self):
        initial = self.candidates[0]
        cost = _fixed_cost(self.data, initial.u, initial.v, initial.w) + solve_scenario(
            self.data, self.scenarios[0], initial.u, initial.v, initial.w).objective
        # At exactly the target cost (reference * 1.005) search must continue.
        self.config.target_cost = cost
        self.config.max_iterations = 1
        for position in (1, 2, 3):
            with self.subTest(position=position):
                answers = [LLMAdviceResult([]) for _ in range(position-1)]
                answers.append(LLMAdviceResult(self.candidates[1:]))
                with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                        "suc_benders.paper_solver.ask_llm_for_candidates", side_effect=answers) as ask, patch(
                        "suc_benders.attribution.validate_schedule", wraps=validate_schedule) as validate:
                    result = solve_benders(self.data, self.scenarios, self.config)
                self.assertEqual(result.status, "TARGET_REACHED")
                self.assertEqual(ask.call_count, position)
                self.assertEqual(result.diagnostics["sa_calls"], 3)
                self.assertEqual(result.diagnostics["evaluation_count"], 2)
                self.assertEqual(validate.call_count, 2)
                np.testing.assert_array_equal(result.u, self.candidates[1].u)
                self.assertLess(result.objective, cost)

    def test_sa_candidate_reaching_target_prevents_llm_dispatch(self):
        from suc_benders.paper_model import key
        candidate = self.candidates[1]
        point = {key(kind, i, t): int(getattr(candidate, kind)[i, t])
                 for kind in ("u", "v", "w") for i in range(3) for t in range(3)}
        self.config.target_cost = 600.
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[[]]), patch(
                "suc_benders.paper_solver.CompactQubo.decode", return_value=(point, None)), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates", side_effect=AssertionError("already reached")):
            result = solve_benders(self.data, self.scenarios, self.config)
        self.assertEqual(result.status, "TARGET_REACHED")
        self.assertEqual(result.diagnostics["sa_calls"], 1)
        self.assertEqual(result.diagnostics["llm_call_count"], 0)
        np.testing.assert_array_equal(result.u, candidate.u)


if __name__ == "__main__":
    unittest.main()
