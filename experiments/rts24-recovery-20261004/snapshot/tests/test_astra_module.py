import json
import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from suc_benders.llm import LLMConfig, LLMAdviceResult, ask_llm_for_candidates
from suc_benders.subagent import initialize, exchange, publish, receipt, claim
from suc_benders.advisor_feedback import summarize, changes, call_due
from suc_benders.data import Scenario
from tests.test_capacity import one_hour_data


class AstraModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        initialize(self.root, {})
        self.config = LLMConfig(enabled=True, api_format="codex_subagent", exchange_dir=str(self.root),
            timeout_seconds=None, collect_diagnostics=True)

    def respond(self, status="completed", corrupt=False):
        call = self.root / "calls/001"
        self.assertTrue((call / "system.txt").exists())
        self.assertTrue((call / "user.txt").exists())
        publish(call / "candidates.json", {"candidates": [{"u": [[1, 1, 1]]}]})
        receipt(self.root, 1, "actual-test-agent", status)
        if corrupt:
            (call / "candidates.json").write_text("{}")

    def test_transport_success_no_http_and_no_reuse(self):
        with patch("suc_benders.subagent.time.sleep", side_effect=lambda _: self.respond()), patch(
                "suc_benders.llm.build_opener", side_effect=AssertionError("HTTP forbidden")):
            data = one_hour_data()
            result = ask_llm_for_candidates(data, [Scenario("tiny", 1., data.base_load)], 1, [], None, self.config)
        self.assertEqual(result.valid_count, 1)
        self.assertFalse(result.failed)
        self.assertEqual(result.diagnostics["http_requests"], 0)
        self.assertTrue(result.candidates[0].provenance["proposal_id"].startswith(result.diagnostics["call_id"]))
        with self.assertRaises(FileExistsError):
            exchange(self.config, 1, "system", "user")
        with self.assertRaises(ValueError):
            receipt(self.root, 1, "again", "completed")

    def test_failed_receipt_is_consumed(self):
        with patch("suc_benders.subagent.time.sleep", side_effect=lambda _: self.respond("failed")):
            content, record = exchange(self.config, 1, "system", "user")
        self.assertIsNone(content)
        self.assertEqual(record["status"], "failed")

    def test_output_integrity_stops(self):
        with patch("suc_benders.subagent.time.sleep", side_effect=lambda _: self.respond(corrupt=True)):
            with self.assertRaisesRegex(ValueError, "integrity"):
                exchange(self.config, 1, "system", "user")

    def test_input_integrity_stops(self):
        def respond(_):
            self.respond()
            (self.root / "calls/001/user.txt").write_text("changed")
        with patch("suc_benders.subagent.time.sleep", side_effect=respond):
            with self.assertRaisesRegex(ValueError, "integrity"):
                exchange(self.config, 1, "system", "user")

    def test_cancel_rejects_ready_result(self):
        def respond(_):
            self.respond()
            publish(self.root / "cancel.json", {})
        with patch("suc_benders.subagent.time.sleep", side_effect=respond):
            content, record = exchange(self.config, 1, "system", "user")
        self.assertIsNone(content)
        self.assertEqual(record["status"], "cancelled")

    def test_deadline_closes_request(self):
        self.config.timeout_seconds = 0
        content, record = exchange(self.config, 1, "system", "user")
        self.assertIsNone(content)
        self.assertEqual(record["status"], "deadline")
        with self.assertRaises(ValueError):
            receipt(self.root, 1, "late", "failed")

    def test_dispatch_claim_is_exclusive(self):
        def respond(_):
            task = claim(self.root, 1)
            self.assertIn("candidates.json", task)
            with self.assertRaises(FileExistsError):
                claim(self.root, 1)
            self.respond()
        with patch("suc_benders.subagent.time.sleep", side_effect=respond):
            exchange(self.config, 1, "system", "user")
        with self.assertRaises(ValueError):
            claim(self.root, 1)

    def test_feedback_costs_repair_duplicates_and_unknown(self):
        proposals = [dict(proposal_id=str(i), commitment_id=str(i), raw_u=[[0, 0, 0]],
                          repaired_u=[[0, 1, 0]], hamming_u=1, outcome="accepted") for i in range(4)]
        records = [dict(commitment_hash=str(i), feasible=i != 2, objective=90+i*20,
                        fixed_cost=40, expected_fuel_cost=50+i*20, improved=i == 0) for i in range(3)]
        baseline = dict(total=100, fixed=50, expected_fuel=50)
        advice = LLMAdviceResult([], diagnostics=dict(proposals=proposals))
        rows = summarize(advice, [[1, 1, 1]], baseline, records, {"1"})["proposals"]
        self.assertEqual(rows[0]["cost_change"]["total"], -10)
        self.assertTrue(rows[0]["refreshed_incumbent"])
        self.assertEqual(rows[0]["repair_changes"], [dict(unit=1, start_hour=2, end_hour=2, **{"from": 0., "to": 1.})])
        self.assertTrue(rows[1]["cache_hit"])
        self.assertEqual(rows[1]["cost_change"]["total"], 10)
        self.assertFalse(rows[2]["feasible"])
        self.assertIsNone(rows[2]["costs"])
        self.assertFalse(rows[3]["validation_completed"])
        self.assertIsNone(rows[3]["feasible"])
        self.assertEqual(changes([[1, 1, 1]], [[0, 0, 1]])[0]["end_hour"], 2)

    def test_two_round_feedback_is_automatic(self):
        from suc_benders.solver import BendersConfig, solve_benders
        from suc_benders.master import QuantumConfig
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        scenarios = [Scenario("tiny", 1., data.base_load)]
        self.config.enhanced_feedback = True
        cfg = BendersConfig(
            max_iterations=2, stop_on_no_new_candidates=False, llm=self.config,
            quantum=QuantumConfig(candidate_pool_size=1))
        def respond(_):
            for call in sorted((self.root / "calls").iterdir()):
                if (call / "receipt.json").exists():
                    continue
                request = json.loads((call / "request.json").read_text())
                publish(call / "candidates.json", {"candidates": [{"u": [[1, 1, 1]]}]})
                receipt(self.root, request["iteration"], "test-agent", "completed")
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.subagent.time.sleep", side_effect=respond), patch(
                "suc_benders.llm.build_opener", side_effect=AssertionError("HTTP forbidden")):
            result = solve_benders(data, scenarios, cfg)
        feedback = json.loads((self.root / "calls/001/feedback.json").read_text())
        second = json.loads((self.root / "calls/002/user.txt").read_text())
        self.assertEqual(second["context"]["previous_call_feedback"], feedback)
        self.assertEqual(result.diagnostics["sa_calls"], 2)
        self.assertTrue(feedback["proposals"][0]["cache_hit"])
        self.assertTrue(feedback["proposals"][0]["validation_completed"])

    def test_same_candidate_has_same_repair_in_http_and_subagent(self):
        data = one_hour_data()
        scenarios = [Scenario("tiny", 1., data.base_load)]
        content = json.dumps({"candidates": [{"u": [[0, 0, 0]]}]})
        response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        class Opener:
            def open(self, *args, **kwargs):
                return io.BytesIO(json.dumps(response).encode())
        with patch.dict("os.environ", {"LLM_API_KEY": "test"}), patch(
                "suc_benders.llm.build_opener", return_value=Opener()):
            legacy = ask_llm_for_candidates(data, scenarios, 1, [], None,
                LLMConfig(api_format="chat_completions", collect_diagnostics=True))
        def respond(_):
            (self.root / "calls/001/candidates.json").write_text(content)
            receipt(self.root, 1, "test-agent", "completed")
        with patch("suc_benders.subagent.time.sleep", side_effect=respond):
            modern = ask_llm_for_candidates(data, scenarios, 1, [], None, self.config)
        self.assertEqual(legacy.valid_count, modern.valid_count)
        self.assertGreater(modern.valid_count, 0)
        for a, b in zip(legacy.candidates, modern.candidates):
            np.testing.assert_array_equal(a.u, b.u)
            np.testing.assert_array_equal(a.v, b.v)
            np.testing.assert_array_equal(a.w, b.w)

    def test_cancel_preserves_completed_best(self):
        from suc_benders.solver import BendersConfig, solve_benders
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        def event(row):
            if row["kind"] == "evaluation":
                publish(self.root / "cancel.json", {})
        cfg = BendersConfig(
            llm=self.config, event_sink=event)
        with patch("suc_benders.paper_solver.solve_qubo", side_effect=AssertionError("cancelled")):
            result = solve_benders(data, [Scenario("tiny", 1., data.base_load)], cfg)
        self.assertEqual(result.status, "CANCELLED")
        self.assertTrue(np.isfinite(result.objective))
        self.assertGreater(result.u.size, 0)

    def test_stagnation_threshold_and_reset(self):
        self.assertTrue(call_due("stagnation", 100., 90., 0, 0)[0])
        stalled = 0
        for expected in (False, False, True):
            due, stalled = call_due("stagnation", 100., 100., stalled, 1)
            self.assertEqual(due, expected)
        due, stalled = call_due("stagnation", 100., 99., stalled, 1)
        self.assertFalse(due)
        self.assertEqual(stalled, 0)
        self.assertTrue(call_due("fixed", 100., 90., 0, 1)[0])

    def test_stagnation_failed_calls_reset_and_limit(self):
        from suc_benders.solver import BendersConfig, solve_benders
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        calls = []
        def fail(*args, **kwargs):
            calls.append(args[2])
            return LLMAdviceResult([], failed=True, message="failed", diagnostics={"proposals": []})
        cfg = BendersConfig(
            max_iterations=8, stop_on_no_new_candidates=False,
            llm=LLMConfig(enabled=True, schedule="stagnation", max_calls=2))
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates", side_effect=fail):
            result = solve_benders(data, [Scenario("tiny", 1., data.base_load)], cfg)
        self.assertEqual(calls, [1, 4])
        self.assertEqual(result.diagnostics["sa_calls"], 8)
        self.assertEqual(result.diagnostics["llm_call_count"], 2)

    def test_advisor_only_has_no_sa_or_integer_optimizer(self):
        import gurobipy as gp
        from suc_benders.solver import BendersConfig, solve_benders
        from suc_benders.candidates import make_candidate
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        scenarios = [Scenario("tiny", 1., data.base_load)]
        candidate = make_candidate(data, scenarios, np.ones((1, 3), dtype=int), source="llm")
        cfg = BendersConfig(
            advisor_only=True, quantum_candidates=False, max_iterations=4,
            stop_on_no_new_candidates=False, llm=LLMConfig(enabled=True, max_calls=2))
        optimize = gp.Model.optimize
        def guard(model, *args, **kwargs):
            model.update()
            self.assertTrue(all(v.VType == gp.GRB.CONTINUOUS for v in model.getVars()))
            return optimize(model, *args, **kwargs)
        with patch("suc_benders.paper_solver.solve_qubo", side_effect=AssertionError("SA forbidden")), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates", return_value=LLMAdviceResult([candidate])), patch.object(gp.Model, "optimize", guard):
            result = solve_benders(data, scenarios, cfg)
        self.assertEqual(result.diagnostics["sa_calls"], 0)
        self.assertEqual(result.diagnostics["llm_call_count"], 2)
        self.assertIsNone(result.diagnostics["reliable_lower"])
        self.assertEqual(result.status, "LLM_CALL_LIMIT")
        from suc_benders.subproblem import solve_scenario
        from suc_benders.solver import _fixed_cost
        expected = solve_scenario(data, scenarios[0], candidate.u, candidate.v, candidate.w).objective
        expected += _fixed_cost(data, candidate.u, candidate.v, candidate.w)
        self.assertAlmostEqual(result.objective, expected)

    def test_target_is_strict_and_overrides_round_limit(self):
        from suc_benders.solver import BendersConfig, solve_benders, _fixed_cost
        from suc_benders.candidates import make_candidate
        from suc_benders.subproblem import solve_scenario
        data = one_hour_data()
        vectors = {k: np.repeat(v, 2) for k, v in vars(data).items()
                   if isinstance(v, np.ndarray) and k != "base_load"}
        data = replace(data, **vectors, n_units=2, base_load=np.full(3, 30.))
        scenarios = [Scenario("tiny", 1., data.base_load)]
        initial = make_candidate(data, scenarios, np.ones((2, 3), dtype=int))
        cost = _fixed_cost(data, initial.u, initial.v, initial.w) + solve_scenario(
            data, scenarios[0], initial.u, initial.v, initial.w).objective
        candidate = make_candidate(data, scenarios, np.array([[1, 1, 1], [0, 0, 0]]), source="llm")
        cfg = BendersConfig(
            max_iterations=1, stop_on_no_new_candidates=False, target_cost=cost,
            llm=LLMConfig(enabled=True))
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates",
                side_effect=[LLMAdviceResult([]), LLMAdviceResult([candidate])]):
            result = solve_benders(data, scenarios, cfg)
        self.assertEqual(result.status, "TARGET_REACHED")
        self.assertLess(result.objective, cost)
        self.assertEqual(result.diagnostics["sa_calls"], 2)


if __name__ == "__main__":
    unittest.main()
