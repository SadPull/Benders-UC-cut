import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import repair_commitment
from suc_benders.llm import LLMConfig, ask_llm_for_candidates
from suc_benders.master import MasterCandidate


ROOT = Path(__file__).resolve().parents[1]


class LLMAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
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
                content = '[{"name":"ok","u":' + json.dumps([[0] * 24 for _ in range(10)]) + '}]'
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


if __name__ == "__main__":
    unittest.main()
