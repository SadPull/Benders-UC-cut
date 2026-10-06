import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import json

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import repair_commitment
from suc_benders.llm import LLMConfig, ask_llm_for_candidates


ROOT = Path(__file__).resolve().parents[1]


class LLMAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")
        cls.base = [
            repair_commitment(
                cls.units,
                cls.scenarios,
                np.zeros((cls.units.n_units, cls.units.horizon), dtype=int),
            )
        ]

    def test_missing_key_is_safe(self):
        with patch.dict("os.environ", {}, clear=True):
            result = ask_llm_for_candidates(
                self.units,
                self.scenarios,
                1,
                self.base,
                None,
                LLMConfig(enabled=True, api_format="messages"),
            )
        self.assertEqual(result.candidates, [])
        self.assertFalse(result.failed)
        self.assertEqual(result.message, "LLM_API_KEY is not set")

    def test_invalid_json_is_rejected(self):
        class Response:
            def read(self):
                return b'{"content":[{"type":"text","text":"{bad"}]}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        class Opener:
            def open(self, *args, **kwargs):
                return Response()

        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch(
            "suc_benders.llm.build_opener", return_value=Opener()
        ):
            result = ask_llm_for_candidates(
                self.units,
                self.scenarios,
                1,
                self.base,
                None,
                LLMConfig(enabled=True, api_format="messages"),
            )
        self.assertTrue(result.failed)
        self.assertEqual(result.candidates, [])
        self.assertNotIn("test-key", result.message)

    def test_list_root_response_is_accepted(self):
        class Response:
            def read(self):
                content = '[{"name":"ok","u":' + json.dumps([[0] * 24 for _ in range(32)]) + '}]'
                return json.dumps({"content":[{"type":"text","text":content}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        class Opener:
            def open(self, *args, **kwargs):
                return Response()

        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch(
            "suc_benders.llm.build_opener", return_value=Opener()
        ):
            result = ask_llm_for_candidates(
                self.units,
                self.scenarios,
                1,
                self.base,
                None,
                LLMConfig(enabled=True, api_format="messages"),
            )
        self.assertFalse(result.failed)
        self.assertEqual(result.raw_count, 1)

    def test_ragged_matrix_is_discarded_not_fatal(self):
        class Response:
            def read(self):
                ragged = [[0, 1], [0, 1, 0]]
                content = json.dumps({"candidates": [{"name": "ragged", "u": ragged}]})
                return json.dumps({"content": [{"type": "text", "text": content}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        class Opener:
            def open(self, *args, **kwargs):
                return Response()

        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch(
            "suc_benders.llm.build_opener", return_value=Opener()
        ):
            result = ask_llm_for_candidates(
                self.units,
                self.scenarios,
                1,
                self.base,
                None,
                LLMConfig(enabled=True, api_format="messages"),
            )
        # Malformed proposal is dropped; the call itself is not a failure.
        self.assertFalse(result.failed)
        self.assertEqual(result.candidates, [])

    def test_invalid_shape_is_discarded(self):
        class Response:
            def read(self):
                content = '{"candidates":[{"name":"bad","u":[[1,2]]}]}'
                return json.dumps({"content":[{"type":"text","text":content}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        class Opener:
            def open(self, *args, **kwargs):
                return Response()

        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}), patch(
            "suc_benders.llm.build_opener", return_value=Opener()
        ):
            result = ask_llm_for_candidates(
                self.units,
                self.scenarios,
                1,
                self.base,
                None,
                LLMConfig(enabled=True, api_format="messages"),
            )
        self.assertFalse(result.failed)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.raw_count, 1)
        self.assertEqual(result.valid_count, 0)

    def responses_call(self, raw, max_tokens=65536):
        response = Mock()
        response.status = 200
        response.read.return_value = json.dumps(raw).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.dict("os.environ", {
            "LLM_API_KEY": "test-key", "ANTHROPIC_AUTH_TOKEN": "other-key",
        }, clear=True), patch("suc_benders.llm.build_opener", return_value=opener):
            result = ask_llm_for_candidates(
                self.units, self.scenarios, 1, [], None,
                LLMConfig(api_format="responses", base_url="https://example.invalid/v1",
                          model="test-model", timeout_seconds=None, max_retries=1,
                          collect_diagnostics=True, max_tokens=max_tokens),
            )
        return result, opener

    def test_responses_payload_and_output_text(self):
        content = json.dumps({"candidates": [{"u": [[0] * 24 for _ in range(32)]}]})
        raw = {"status": "completed", "model": "test-model",
               "usage": {"input_tokens": 5, "output_tokens": 10}, "output": [
                   {"type": "reasoning", "summary": [{"text": "not candidate JSON"}]},
                   {"type": "message", "role": "assistant", "content": [
                       {"type": "output_text", "text": content[:20]},
                       {"type": "output_text", "text": content[20:]},
                   ]},
               ]}
        result, opener = self.responses_call(raw)
        self.assertFalse(result.failed)
        self.assertEqual(result.valid_count, 1)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual([x["role"] for x in payload["input"]], ["system", "user"])
        self.assertEqual(payload["max_output_tokens"], 65536)
        self.assertEqual(payload["reasoning"], {"effort": "max"})
        self.assertNotIn("messages", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["store"])
        self.assertIsNone(opener.open.call_args.kwargs["timeout"])
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(result.diagnostics["usage"], raw["usage"])
        self.assertNotIn("test-key", json.dumps(result.diagnostics))

    def test_responses_can_use_provider_output_limit(self):
        raw = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"candidates":[]}'}]}]}
        result, opener = self.responses_call(raw, max_tokens=None)
        self.assertFalse(result.failed)
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertNotIn("max_output_tokens", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(opener.open.call_count, 1)

    def test_responses_rejects_incomplete_even_with_parseable_text(self):
        for reason, message in (("max_output_tokens", "truncated"),
                                ("content_filter", "response_incomplete")):
            with self.subTest(reason=reason):
                raw = {"status": "incomplete", "incomplete_details": {"reason": reason},
                       "output": [{"type": "message", "content": [
                           {"type": "output_text", "text": '{"candidates":[]}'},
                       ]}]}
                result, opener = self.responses_call(raw)
                self.assertTrue(result.failed)
                self.assertEqual(result.message, message)
                self.assertEqual(result.candidates, [])
                self.assertEqual(opener.open.call_count, 1)

    def test_responses_failure_empty_and_invalid_json(self):
        cases = [
            ({"status": "failed", "error": {"message": "test-key"}}, "response_failed"),
            ({"status": "completed", "output": []}, "empty_content"),
            ({"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "{bad"}]}]}, "invalid_candidate_json"),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                result, opener = self.responses_call(raw)
                self.assertTrue(result.failed)
                self.assertEqual(result.message, message)
                self.assertEqual(opener.open.call_count, 1)
                self.assertNotIn("test-key", json.dumps(result.diagnostics))


if __name__ == "__main__":
    unittest.main()
