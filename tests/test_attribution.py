import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import numpy as np

from suc_benders import attribution
from suc_benders.attribution_report import checkpoint, describe, generate_report, paired_difference, paired_effects
from suc_benders.candidates import make_candidate
from suc_benders.data import Scenario
from suc_benders.full_model import FullResult
from suc_benders.llm import LLMAdviceResult, LLMConfig, ask_llm_for_candidates
from suc_benders.master import QuantumConfig
from suc_benders.solver import BendersConfig, result_to_dict, solve_benders
from suc_benders.subproblem import solve_scenario
from tests.test_capacity import one_hour_data


def fixture():
    data = replace(one_hour_data(), pmin=np.array([0.]))
    scenarios = [Scenario(name, probability, data.base_load * factor) for name, probability, factor
                 in (("low", .25, .9), ("base", .5, 1.), ("high", .25, 1.05))]
    return data, scenarios


def response(content='{"candidates":[{"u":[[1,0,0]]}]}', finish="stop"):
    value = {
        "model": "test-model", "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
    }
    result = Mock()
    result.status = 200
    result.read.return_value = json.dumps(value).encode()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class ChatCompletionsTests(unittest.TestCase):
    def setUp(self):
        self.data, self.scenarios = fixture()
        self.config = LLMConfig(
            api_format="chat_completions", base_url="https://example.invalid/v1/",
            model="test-model", max_tokens=8192, max_retries=2, collect_diagnostics=True,
        )

    def call(self, opener):
        with patch.dict("os.environ", {"LLM_API_KEY": "test-audit-key"}, clear=True), \
             patch("suc_benders.llm.build_opener", return_value=opener):
            return ask_llm_for_candidates(self.data, self.scenarios, 1, [], None, self.config)

    def test_chat_payload_and_safe_audit(self):
        opener = Mock()
        opener.open.return_value = response()
        result = self.call(opener)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-audit-key")
        payload = json.loads(request.data)
        self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertFalse(payload["stream"])
        self.assertFalse(result.failed)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.candidates[0].provenance["origin"], "llm")
        self.assertEqual(result.diagnostics["usage"]["completion_tokens"], 10)
        self.assertNotIn("test-audit-key", json.dumps(result.diagnostics))
        self.assertNotIn("Authorization", json.dumps(result.diagnostics))

    def test_transient_http_retries_once_and_scrubs_echoed_key(self):
        opener = Mock()
        opener.open.side_effect = [
            HTTPError("https://example.invalid", 503, "unavailable", {}, io.BytesIO(b"test-audit-key")),
            response(),
        ]
        with patch("suc_benders.llm.time.sleep") as sleep:
            result = self.call(opener)
        self.assertFalse(result.failed)
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(30.)
        self.assertEqual([row["status_code"] for row in result.diagnostics["attempts"]], [503, 200])
        self.assertNotIn("test-audit-key", json.dumps(result.diagnostics))

    def test_auth_failure_is_not_retried(self):
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.invalid", 401, "unauthorized", {}, io.BytesIO(b"test-audit-key"))
        with patch("suc_benders.llm.time.sleep") as sleep:
            result = self.call(opener)
        self.assertTrue(result.failed)
        self.assertEqual(opener.open.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(result.message, "HTTP 401")
        self.assertNotIn("test-audit-key", json.dumps(result.diagnostics))

    def test_network_failure_stops_at_attempt_limit(self):
        opener = Mock()
        opener.open.side_effect = URLError("unavailable")
        with patch("suc_benders.llm.time.sleep"):
            result = self.call(opener)
        self.assertTrue(result.failed)
        self.assertEqual(opener.open.call_count, 2)

    def test_truncated_empty_and_invalid_json_are_distinct(self):
        for content, finish, message in (
            ('{"candidates":[]}', "length", "truncated"),
            ("", "stop", "empty_content"),
            ("{", "stop", "invalid_candidate_json"),
        ):
            with self.subTest(message=message):
                opener = Mock()
                opener.open.return_value = response(content, finish)
                result = self.call(opener)
                self.assertTrue(result.failed)
                self.assertEqual(result.message, message)
                self.assertEqual(result.candidates, [])
                self.assertEqual(len(result.diagnostics["attempts"]), 1)

    def test_duplicate_and_nonbinary_proposals_are_recorded(self):
        opener = Mock()
        opener.open.return_value = response('{"candidates":[{"u":[[1,0,0]]},{"u":[[1,0,0]]},{"u":[[0.9,0,0]]}]}')
        result = self.call(opener)
        self.assertFalse(result.failed)
        self.assertEqual(result.raw_count, 3)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual([row["outcome"] for row in result.diagnostics["proposals"]], ["accepted", "duplicate", "duplicate"])
        self.assertFalse(result.diagnostics["proposals"][-1]["raw_binary"])


class EvaluationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.data, self.scenarios = fixture()

    def config(self, **changes):
        return BendersConfig(**{
            "quantum_candidates": False, "decommit_candidates": False,
            "max_iterations": 3, "collect_diagnostics": True, **changes,
        })

    def test_budget_counts_seed_and_prevents_llm_and_decommit(self):
        with patch("suc_benders.solver.ask_llm_for_candidates", side_effect=AssertionError("LLM called")), \
             patch("suc_benders.solver.decommit_commitment", side_effect=AssertionError("decommit called")), \
             patch("suc_benders.solver.solve_scenario", wraps=solve_scenario) as solve:
            result = solve_benders(self.data, self.scenarios, self.config(
                max_candidate_evaluations=1, llm=LLMConfig(enabled=True), decommit_candidates=True,
            ))
        self.assertEqual(result.status, "EVALUATION_LIMIT")
        self.assertEqual(solve.call_count, len(self.scenarios))
        self.assertEqual(result.diagnostics["evaluation_count"], 1)
        self.assertEqual(result.diagnostics["llm_calls"], [])
        self.assertTrue(attribution.validate_result(self.data, self.scenarios, result)["feasible_solution"])

    def test_budget_can_end_inside_llm_batch(self):
        proposals = [make_candidate(self.data, self.scenarios, np.array([row]), source="llm")
                     for row in ([1, 0, 0], [1, 0, 1])]
        with patch("suc_benders.solver.ask_llm_for_candidates", return_value=LLMAdviceResult(proposals, 2, 2)) as ask, \
             patch("suc_benders.solver.solve_scenario", wraps=solve_scenario) as solve:
            result = solve_benders(self.data, self.scenarios, self.config(
                max_candidate_evaluations=2, llm=LLMConfig(enabled=True),
            ))
        self.assertEqual(result.status, "EVALUATION_LIMIT")
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(solve.call_count, 2 * len(self.scenarios))
        self.assertEqual(result.diagnostics["evaluation_count"], 2)
        np.testing.assert_array_equal(result.u, [[1, 0, 0]])

    def test_duplicates_do_not_use_budget_or_receive_credit(self):
        same = make_candidate(self.data, self.scenarios, np.array([[1, 1, 1]]), source="llm")
        better = make_candidate(self.data, self.scenarios, np.array([[1, 0, 0]]), source="llm")
        with patch("suc_benders.solver.ask_llm_for_candidates",
                   return_value=LLMAdviceResult([same, same, better, better], 4, 4)), \
             patch("suc_benders.solver.solve_scenario", wraps=solve_scenario) as solve:
            result = solve_benders(self.data, self.scenarios, self.config(
                max_iterations=1, max_candidate_evaluations=3, llm=LLMConfig(enabled=True),
            ))
        events = result.diagnostics["evaluations"]
        self.assertEqual(result.diagnostics["evaluation_count"], 2)
        self.assertEqual(solve.call_count, 2 * len(self.scenarios))
        self.assertEqual(sum(row["cache_hit"] for row in events), 3)
        self.assertTrue(all(not row["improved"] and row["improvement"] == 0 for row in events if row["cache_hit"]))
        self.assertEqual([row["origin"] for row in events if not row["cache_hit"]], ["fallback", "llm"])

    def test_quantum_origin_survives_transition_repair(self):
        raw = make_candidate(self.data, self.scenarios, np.array([[1, 1, 1]]))
        raw.v[:] = 0
        master = Mock()
        master.matrix.return_value = np.eye(9)
        master.energy_unit = 1.
        master.variable_count.return_value = 9
        master.decode.side_effect = [copy.deepcopy(raw), copy.deepcopy(raw)]
        with patch("suc_benders.solver.QuantumMaster", return_value=master), \
             patch("suc_benders.solver.solve_qubo", return_value=[np.ones(9), np.ones(9)]):
            result = solve_benders(self.data, self.scenarios, self.config(
                quantum_candidates=True, max_iterations=1,
            ))
        proposals = [row for row in result.diagnostics["proposals"] if "raw_id" in row]
        self.assertEqual(len(proposals), 2)
        self.assertTrue(all(not row["raw_discrete_feasible"] for row in proposals))
        self.assertTrue(all(row["hamming_u"] == 0 and row["hamming_uvw"] > 0 for row in proposals))
        event = result.diagnostics["evaluations"][0]
        self.assertEqual(event["source"], "repair")
        self.assertEqual(event["origin"], "quantum")
        self.assertEqual(result.diagnostics["evaluation_count"], 1)

    def test_diagnostics_do_not_change_unlimited_annealing(self):
        config = self.config(
            quantum_candidates=True, max_iterations=2, collect_diagnostics=False,
            quantum=QuantumConfig(kaiwu_seed=17, candidate_pool_size=4, initial_temperature=20,
                                  alpha=.9, cutoff_temperature=.1, iterations_per_temperature=2),
        )
        plain = solve_benders(self.data, self.scenarios, config)
        traced = solve_benders(self.data, self.scenarios, replace(config, collect_diagnostics=True))
        self.assertEqual(plain.diagnostics, {})
        self.assertEqual(plain.status, traced.status)
        self.assertAlmostEqual(plain.objective, traced.objective, places=7)
        np.testing.assert_array_equal(plain.u, traced.u)
        self.assertEqual([asdict(row) for row in plain.history], [asdict(row) for row in traced.history])

    def test_random_control_is_seeded_and_skips_kaiwu(self):
        config = self.config(random_candidates=True, random_seed=27, max_iterations=1, max_candidate_evaluations=2)
        with patch("suc_benders.solver.QuantumMaster", side_effect=AssertionError("QUBO called")), \
             patch("suc_benders.solver.solve_qubo", side_effect=AssertionError("SA called")):
            first = solve_benders(self.data, self.scenarios, config)
            second = solve_benders(self.data, self.scenarios, config)
        raw = lambda result: [row["raw_u"].tolist() for row in result.diagnostics["proposals"] if "raw_u" in row]
        self.assertEqual(raw(first), raw(second))
        self.assertEqual(len(raw(first)), 16)
        self.assertEqual(first.diagnostics["sampling"], [])

    def test_invalid_budget_and_mixed_generator_are_rejected(self):
        for config in (self.config(max_candidate_evaluations=0), self.config(random_candidates=True, quantum_candidates=True)):
            with self.assertRaises(ValueError):
                solve_benders(self.data, self.scenarios, config)

    def test_validator_detects_tampered_dispatch(self):
        result = solve_benders(self.data, self.scenarios, self.config(max_candidate_evaluations=1))
        attribution.validate_result(self.data, self.scenarios, result)
        result.scenarios[0].p[0, 0] += 1.
        with self.assertRaises(AssertionError):
            attribution.validate_result(self.data, self.scenarios, result)


class AttributionDesignTests(unittest.TestCase):
    def test_factorial_schedule_and_call_limit(self):
        rows = attribution.schedule()
        self.assertEqual(rows, attribution.schedule())
        self.assertEqual(len(rows), 150)
        self.assertEqual(len({row["run_id"] for row in rows}), 150)
        self.assertEqual(sum(row["strategy"]["l"] for row in rows) * 10, 600)
        self.assertEqual({(row["case"], row["seed"]) for row in rows[:10]}, {("canonical", 7)})
        for start in range(0, len(rows), 10):
            self.assertEqual({row["strategy"]["name"] for row in rows[start:start+10]},
                             {strategy["name"] for strategy in attribution.STRATEGIES})

    def test_case_transformations_preserve_probabilities_and_peak(self):
        data, scenarios = fixture()
        light = attribution.case_scenarios(scenarios, "light_90")
        shifted = attribution.case_scenarios(scenarios, "peak_shift_2h")
        for before, a, b in zip(scenarios, light, shifted):
            np.testing.assert_allclose(a.load, before.load * .9)
            np.testing.assert_array_equal(b.load, np.roll(before.load, 2))
            self.assertEqual(a.probability, before.probability)
            self.assertEqual(b.load.max(), before.load.max())

    def test_manifest_rejects_changed_settings_on_resume(self):
        data, scenarios = fixture()
        config = LLMConfig(base_url="https://example.invalid/v1", model="test-model")
        with tempfile.TemporaryDirectory() as directory, \
             patch("suc_benders.attribution.platform.platform", return_value="test-platform"), \
             patch("suc_benders.attribution.subprocess.check_output", return_value=b"snapshot\n"):
            directory = Path(directory)
            path = attribution.ROOT / "data/suc_10x24.json"
            first = attribution.prepare_manifest(directory, path, data, scenarios, config, False)
            again = attribution.prepare_manifest(directory, path, data, scenarios, config, True)
            self.assertEqual(first["fingerprint"], again["fingerprint"])
            with self.assertRaises(ValueError):
                attribution.prepare_manifest(directory, path, data, scenarios, replace(config, model="changed"), True)
            with patch.dict("os.environ", {"OMP_NUM_THREADS": "99"}), self.assertRaises(ValueError):
                attribution.prepare_manifest(directory, path, data, scenarios, config, True)

    def test_pilot_resume_and_reference_order(self):
        data, scenarios = fixture()
        result = solve_benders(data, scenarios, BendersConfig(
            quantum_candidates=False, decommit_candidates=False, max_iterations=1,
            max_candidate_evaluations=1, collect_diagnostics=True,
        ))
        events = []

        def research(*args):
            events.append("research")
            return copy.deepcopy(result)

        def reference(*args):
            events.append("reference")
            return FullResult("OPTIMAL", 1., 1., 0., result.u, result.v, result.w, [], 0., 1.,
                              model=Mock(), lower_bound=1.)

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"LLM_API_KEY": "test-suite-key"}), \
             patch("suc_benders.attribution.load_suc_data", return_value=(data, scenarios)), \
             patch("suc_benders.attribution.require_gurobi", return_value=Mock()), \
             patch("suc_benders.attribution.platform.platform", return_value="test-platform"), \
             patch("suc_benders.attribution.validate_result", return_value={"feasible_solution": True, "trace_valid": True}), \
             patch("suc_benders.attribution.solve_benders", side_effect=research), \
             patch("suc_benders.attribution.solve_full", side_effect=reference), \
             patch("suc_benders.attribution.subprocess.check_output", return_value=b"snapshot\n"), \
             patch("suc_benders.attribution_report.generate_report") as report, \
             contextlib.redirect_stdout(io.StringIO()):
            kwargs = dict(data_path=attribution.ROOT / "data/suc_10x24.json", output_dir=directory,
                          base_url="https://example.invalid/v1", model="test-model")
            attribution.run_attribution(**kwargs, pilot=True)
            self.assertEqual(events, ["research"] * 10)
            report.assert_not_called()
            attribution.run_attribution(**kwargs, resume=True)
            self.assertEqual(events, ["research"] * 150 + ["reference"] * 3)
            self.assertEqual(report.call_count, 1)
            attribution.run_attribution(**kwargs, resume=True)
            self.assertEqual(len(events), 153)
            self.assertEqual(report.call_count, 2)
            with patch("suc_benders.attribution_report.make_figures"):
                exported = generate_report(directory)
            self.assertEqual(exported, {"runs": 150, "llm_calls": 0, "llm_requests": 0})
            self.assertTrue((Path(directory) / "effects.csv").is_file())
            self.assertIn("150", (Path(directory) / "report.md").read_text(encoding="utf-8"))


class AttributionStatisticsTests(unittest.TestCase):
    def test_control_difference_is_paired_and_excludes_missing_values(self):
        result = paired_difference([100., None, 80.], [90., 200., 70.])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['mean'], 10.)
        self.assertEqual(result['ci_low'], 10.)
        self.assertEqual(result['ci_high'], 10.)
        self.assertIsNone(paired_difference([None], [1.])['mean'])

    def test_checkpoint_never_looks_ahead_and_holds_stopped_incumbent(self):
        events = [
            {"evaluation_count": 1, "elapsed_seconds": 5., "cache_hit": False, "incumbent": 100.},
            {"evaluation_count": 2, "elapsed_seconds": 20., "cache_hit": False, "incumbent": 90.},
            {"evaluation_count": 2, "elapsed_seconds": 25., "cache_hit": True, "incumbent": 90.},
        ]
        self.assertIsNone(checkpoint(events, "seconds", 4.))
        self.assertEqual(checkpoint(events, "seconds", 10.), 100.)
        self.assertEqual(checkpoint(events, "seconds", 30.), 90.)
        self.assertEqual(checkpoint(events, "evaluations", 40), 90.)
        self.assertEqual(checkpoint(events, "evaluations", 1), 100.)

    def test_paired_interaction_sign_and_bootstrap(self):
        costs = [[100., 90., 80., 60.]] * 5
        effects = {row["effect"]: row for row in paired_effects(costs)}
        self.assertEqual(effects["interaction"]["mean"], 10.)
        self.assertEqual(effects["interaction"]["ci_low"], 10.)
        self.assertEqual(effects["interaction"]["ci_high"], 10.)
        self.assertEqual(effects["q_gain_with_l"]["mean"], 20.)
        self.assertEqual(effects["l_gain_with_q"]["mean"], 30.)
        self.assertEqual(paired_effects(costs), paired_effects(costs))

    def test_missing_pairs_are_reported_not_replaced_by_zero(self):
        costs = [[100., 90., 80., 60.], [100., np.nan, 80., 60.]]
        effects = paired_effects(costs)
        self.assertTrue(all(row["n"] == 1 and row["ci_low"] is None for row in effects))
        self.assertEqual(describe([None, 2., 4.]), {"n": 2, "mean": 3., "median": 3., "std": np.sqrt(2.)})
        self.assertEqual(describe([None])["n"], 0)


if __name__ == "__main__":
    unittest.main()
