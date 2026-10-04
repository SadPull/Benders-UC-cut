import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from suc_benders import astra_run, full
from suc_benders.solver import BendersConfig, BendersResult


def research_result(objective=120.0):
    return BendersResult(
        "MAX_ITERATIONS", objective, np.ones((1, 1), dtype=int),
        np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int), [], [], 0, 0, 2.0,
    )


class InterfaceTests(unittest.TestCase):
    def test_package_entry_uses_current_runner(self):
        from suc_benders.__main__ import main
        self.assertIs(main, astra_run.main)

    def test_default_input_and_strategies(self):
        for strategy in ("sa", "fixed", "stagnation", "stagnation_burst", "llm"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                with patch("sys.argv", ["suc_benders", "--output-dir", str(root), "--strategy", strategy]), \
                     patch.object(astra_run, "solve_benders", return_value=research_result()) as solve:
                    astra_run.main()
                data, scenarios, config = solve.call_args.args
                self.assertEqual((data.n_units, data.horizon), (32, 24))
                self.assertEqual(len(scenarios), 3)
                self.assertEqual(config.quantum_candidates, strategy != "llm")
                self.assertEqual(config.llm.enabled, strategy != "sa")
                self.assertEqual(config.advisor_only, strategy == "llm")
                self.assertEqual(config.llm.api_format, "codex_subagent")
                self.assertEqual(json.loads((root / "result.json").read_text())["objective"], 120.0)

    def test_retired_options_are_rejected_before_running(self):
        for args in (["--master-encoding", "legacy"], ["--decomposition", "temporal"],
                     ["--no-decommit"], ["--suite", "attribution"]):
            with self.subTest(args=args), patch("sys.argv", ["suc_benders", "--output-dir", "unused", *args]), \
                 patch.object(astra_run, "solve_benders") as solve, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                astra_run.main()
            self.assertEqual(raised.exception.code, 2)
            solve.assert_not_called()
        for key in ("master_encoding", "decomposition", "decommit_candidates", "random_candidates"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                BendersConfig(**{key: True})

    def test_full_cli_uses_rts24_and_reports_reference_bound(self):
        result = SimpleNamespace(status="TIME_LIMIT", objective=100., lower_bound=90.,
            fixed_cost=20., expected_fuel_cost=80., mip_gap=.1, runtime=1.)
        with patch("sys.argv", ["full", "--quiet", "--time-limit", "1"]), \
             patch.object(full, "solve_full", return_value=result) as solve, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            full.main()
        data, scenarios, config = solve.call_args.args
        self.assertEqual((data.n_units, data.horizon, len(scenarios)), (32, 24, 3))
        self.assertEqual(config.time_limit, 1.)
        self.assertIn("objective=100.000000", output.getvalue())
        self.assertIn("lower_bound=90.000000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
