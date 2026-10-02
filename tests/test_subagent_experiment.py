import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from suc_benders.attribution import write_json
from suc_benders.data import Scenario
from suc_benders.llm import _prompt
from suc_benders.subagent_experiment import adviser, configuration, sha, MODEL
from tests.test_capacity import one_hour_data


class SubagentExperimentTests(unittest.TestCase):
    def test_original_prompt_parser_and_one_receipt_without_http(self):
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        scenarios = [Scenario("tiny", 1., data.base_load)]
        config = configuration(7, "sa_llm").llm
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            call = path / "calls/001"
            expected = _prompt(data, scenarios, 1, None, [], 4, {"example": 1})
            def respond(_):
                self.assertEqual((call / "system.txt").read_text(), expected[0])
                self.assertEqual((call / "user.txt").read_text(), expected[1])
                request = json.loads((call / "request.json").read_text())
                write_json(call / "candidates.json", {"candidates": [{"u": [[1, 1, 1]]}]})
                write_json(call / "receipt.json", {"model": MODEL, "reasoning_effort": "max",
                    "input_sha256": request["input_sha256"], "output_sha256": sha(call / "candidates.json"),
                    "subagent_calls": 1, "agent_id": "test-only"})
            with patch("suc_benders.subagent_experiment.time.sleep", side_effect=respond) as wait, \
                 patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP forbidden")):
                result = adviser(path, 7)(data, scenarios, 1, [], None, config, context={"example": 1})
            self.assertFalse(result.failed)
            self.assertEqual(result.valid_count, 1)
            self.assertEqual(wait.call_count, 1)
            self.assertEqual(result.diagnostics["http_requests"], 0)
            self.assertTrue(result.diagnostics["synthetic_envelope"])
            self.assertTrue(np.array_equal(result.candidates[0].u, [[1, 1, 1]]))
            with self.assertRaises(FileExistsError):
                # Repeating a consumed round must never invoke another agent.
                from suc_benders.subagent_experiment import SubagentTransport
                from urllib.request import Request
                payload = {"model": MODEL, "reasoning": {"effort": "max"},
                           "input": [{"content": s} for s in expected]}
                SubagentTransport(path, 7).open(Request("https://offline.invalid", data=json.dumps(payload).encode()))

    def test_modified_candidate_is_rejected(self):
        data = replace(one_hour_data(), base_load=np.full(3, 30.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            call = path / "calls/001"
            def respond(_):
                request = json.loads((call / "request.json").read_text())
                write_json(call / "candidates.json", {"candidates": []})
                write_json(call / "receipt.json", {"model": MODEL, "reasoning_effort": "max",
                    "input_sha256": request["input_sha256"], "output_sha256": "incorrect",
                    "subagent_calls": 1, "agent_id": "test-only"})
            with patch("suc_benders.subagent_experiment.time.sleep", side_effect=respond), \
                 self.assertRaises(AssertionError):
                adviser(path, 7)(data, [Scenario("tiny", 1., data.base_load)], 1, [], None,
                                 configuration(7, "sa_llm").llm)

    def test_fixed_rounds_and_unchanged_sa_configuration(self):
        from suc_benders.efficiency_experiment import configuration as original
        self.assertEqual(configuration(17, "sa"), original(17, "sa", rounds=50))
        config = configuration(17, "sa_llm")
        self.assertEqual(config.max_iterations, 50)
        self.assertIsNone(config.time_limit_seconds)
        self.assertIsNone(config.llm.timeout_seconds)
        self.assertIsNone(config.llm.max_tokens)
        self.assertEqual(config.llm.model, MODEL)
        self.assertEqual(config.llm.max_retries, 1)
        self.assertEqual(config.llm.reasoning_effort, "max")


if __name__ == "__main__":
    unittest.main()
