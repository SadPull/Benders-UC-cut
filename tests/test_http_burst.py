import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from suc_benders import astra_run
from suc_benders.data import Scenario
from suc_benders.llm import LLMConfig, ask_llm_for_candidates
from tests.test_capacity import one_hour_data
from tests.test_interfaces import research_result


class HTTPBurstTests(unittest.TestCase):
    def test_optional_sampling_parameters_and_redacted_diagnostics(self):
        data = one_hour_data()
        payloads = []
        class Opener:
            def open(self, request, timeout):
                payloads.append(json.loads(request.data))
                self_test.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
                self_test.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
                self_test.assertIsNone(timeout)
                saved = Path(temporary.name) / "calls/001/request.json"
                self_test.assertEqual(json.loads(saved.read_text()), payloads[-1])
                self_test.assertTrue((saved.parent / "system.txt").exists())
                self_test.assertTrue((saved.parent / "user.txt").exists())
                return io.BytesIO(json.dumps(dict(model="deepseek-flash", echoed="test-secret",
                    choices=[dict(message=dict(content='{"candidates":[]}'), finish_reason="stop")],
                    usage=dict(total_tokens=42))).encode())
        self_test = self
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = LLMConfig(api_format="chat_completions", base_url="https://api.deepseek.com",
            model="deepseek-flash", temperature=None, max_tokens=None, timeout_seconds=None,
            max_retries=1, collect_diagnostics=True, exchange_dir=temporary.name)
        with patch.dict("os.environ", {"LLM_API_KEY": "test-secret"}), patch(
                "suc_benders.llm.build_opener", return_value=Opener()):
            result = ask_llm_for_candidates(data, [Scenario("tiny", 1., data.base_load)], 1, [], None, config)
        self.assertNotIn("max_tokens", payloads[0])
        self.assertNotIn("temperature", payloads[0])
        self.assertEqual(payloads[0]["reasoning_effort"], "max")
        self.assertEqual(result.diagnostics["usage"], {"total_tokens": 42})
        self.assertNotIn("test-secret", json.dumps(result.diagnostics))

    def test_cli_preserves_burst_settings_and_saves_http_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            args = ["astra_run", "--output-dir", str(root), "--strategy", "stagnation_burst",
                    "--http-base-url", "https://api.deepseek.com", "--http-model", "deepseek-flash"]
            def solve(data, scenarios, config):
                self.assertEqual(config.llm.schedule, "stagnation_burst")
                self.assertEqual(config.llm.api_format, "chat_completions")
                self.assertEqual(config.llm.model, "deepseek-flash")
                self.assertEqual(config.llm.max_retries, 1)
                self.assertTrue(config.llm.enhanced_feedback)
                self.assertIsNone(config.llm.max_tokens)
                self.assertIsNone(config.llm.timeout_seconds)
                self.assertIsNone(config.llm.temperature)
                self.assertEqual(config.quantum.kaiwu_seed, 7)
                self.assertEqual(config.quantum.candidate_pool_size, 16)
                request = dict(messages=[dict(role="system", content="system"), dict(role="user", content="user")])
                config.event_sink(dict(kind="llm_start", call_index=1, iteration=3, batch_index=1))
                config.event_sink(dict(kind="llm_end", diagnostics=dict(request=request, response={"id": "test"})))
                config.event_sink(dict(kind="llm_feedback", feedback={"proposals": []}))
                return research_result()
            with patch.dict("os.environ", {"LLM_API_KEY": "test-secret"}), patch(
                    "sys.argv", args), patch.object(astra_run, "solve_benders", side_effect=solve):
                astra_run.main()
            call = root / "calls/001"
            self.assertEqual((call / "system.txt").read_text(), "system")
            self.assertEqual(json.loads((call / "response.json").read_text()), {"id": "test"})
            self.assertEqual(json.loads((call / "feedback.json").read_text()), {"proposals": []})
            self.assertNotIn("test-secret", (root / "run.json").read_text())

    def test_incomplete_http_settings_fail_before_starting(self):
        for extra in (["--http-model", "deepseek-flash"],
                      ["--http-base-url", "https://api.deepseek.com"],
                      ["--http-base-url", "https://api.deepseek.com", "--http-model", "deepseek-flash"]):
            with patch.dict("os.environ", {}, clear=True), patch("sys.argv", ["astra_run", "--output-dir", "unused", *extra]), patch.object(
                    astra_run, "initialize", side_effect=AssertionError("must reject before starting")), self.assertRaises(SystemExit):
                astra_run.main()


if __name__ == "__main__":
    unittest.main()
