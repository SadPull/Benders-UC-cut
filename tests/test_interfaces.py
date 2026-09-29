import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from suc_benders import benchmark, full, run
from suc_benders.solver import BendersConfig, BendersResult, solve_benders
from suc_benders.subproblem import solve_scenario


ROOT = Path(__file__).resolve().parents[1]


def research_result(objective=120.0):
    return BendersResult(
        "MAX_ITERATIONS", objective, np.ones((1, 1), dtype=int),
        np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int), [], [], 0, 0, 2.0,
    )


def reference_result(objective=100.0, lower_bound=90.0):
    return SimpleNamespace(
        objective=objective, lower_bound=lower_bound, status="TIME_LIMIT",
        mip_gap=0.1, runtime=999.0, fixed_cost=20.0, expected_fuel_cost=80.0,
    )


class InterfaceTests(unittest.TestCase):
    def test_research_cli_defaults_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with patch("sys.argv", ["run", "--output", str(path)]), \
                 patch.object(run, "load_suc_data", return_value=(None, [])), \
                 patch.object(run, "solve_benders", return_value=research_result()) as solve, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                run.main()
            config = solve.call_args.args[2]
            self.assertTrue(config.quantum_candidates)
            self.assertTrue(config.decommit_candidates)
            self.assertFalse(config.llm.enabled)
            payload = json.loads(path.read_text())
        self.assertEqual(payload["objective"], 120.0)
        for removed in ("lower_bound", "gap", "master_status", "master_runtime"):
            self.assertNotIn(removed, payload)
            self.assertNotIn(removed, output.getvalue())

    def test_research_cli_ablation_flags(self):
        with patch("sys.argv", ["run", "--no-quantum", "--no-decommit", "--llm"]), \
             patch.object(run, "load_suc_data", return_value=(None, [])), \
             patch.object(run, "solve_benders", return_value=research_result()) as solve, \
             contextlib.redirect_stdout(io.StringIO()):
            run.main()
        config = solve.call_args.args[2]
        self.assertFalse(config.quantum_candidates)
        self.assertFalse(config.decommit_candidates)
        self.assertTrue(config.llm.enabled)

    def test_obsolete_cli_parameters_are_rejected(self):
        for module, args in (
            (run, ["--no-exact-master"]), (run, ["--master-time-limit", "1"]),
            (run, ["--master-mip-gap", "0.1"]), (run, ["--no-perturb"]),
            (benchmark, ["--master-time-limit", "1"]),
        ):
            with self.subTest(module=module.__name__, args=args), \
                 patch("sys.argv", [module.__name__, *args]), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                module.main()
            self.assertEqual(raised.exception.code, 2)
        for key in ("exact_master", "exact_master_config", "gap_tolerance", "perturb_candidates",
                    "perturbations_per_candidate", "perturbation_window_lengths", "group_size",
                    "max_generated_candidates"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                BendersConfig(**{key: True})

    def benchmark_payload(self, reference, result=None, args=()):
        events = []

        def solve_research(*unused):
            events.append("research")
            return result if result is not None else research_result()

        def solve_reference(*unused):
            events.append("reference")
            return reference

        with patch("sys.argv", ["benchmark", *args]), \
             patch.object(benchmark, "load_suc_data", return_value=(None, [])), \
             patch.object(benchmark, "solve_benders", side_effect=solve_research), \
             patch.object(benchmark, "solve_full", side_effect=solve_reference), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            benchmark.main()
        return json.loads(output.getvalue()), events

    def test_reference_is_posthoc_and_separately_reported(self):
        payload, events = self.benchmark_payload(reference_result(), args=["--llm"])
        self.assertEqual(events, ["research"] * 5 + ["reference"])
        self.assertEqual({row["strategy"] for row in payload["rows"]}, {
            "quantum-only", "quantum+repair", "quantum+repair+decommit",
            "quantum+repair+llm", "quantum+repair+decommit+llm",
        })
        self.assertEqual(payload["exact_status"], "TIME_LIMIT")
        self.assertEqual(payload["exact_lower_bound"], 90.0)
        self.assertEqual(payload["exact_objective"], 100.0)
        self.assertEqual(payload["exact_mip_gap"], 0.1)
        self.assertEqual(payload["exact_mip_gap_tolerance"], 1e-4)
        self.assertEqual(payload["exact_runtime"], 999.0)
        for row in payload["rows"]:
            self.assertEqual(row["relative_error"], 0.2)
            self.assertEqual(row["reference_gap"], 0.25)
            self.assertEqual(row["runtime"], 2.0)
            self.assertFalse({"lower_bound", "benders_gap", "exact_candidates", "perturbed_candidates"} & row.keys())

    def test_unavailable_comparisons_serialize_as_null(self):
        cases = [
            (reference_result(float("inf"), float("-inf")), 120.0, None, None),
            (reference_result(0.0, float("-inf")), 120.0, None, None),
            (reference_result(), float("inf"), None, None),
            (reference_result(float("inf"), 90.0), 120.0, None, 0.25),
        ]
        for reference, objective, error, gap in cases:
            with self.subTest(reference=reference, objective=objective):
                payload, _ = self.benchmark_payload(reference, research_result(objective))
                self.assertEqual(payload["rows"][0]["relative_error"], error)
                self.assertEqual(payload["rows"][0]["reference_gap"], gap)

    def test_removed_benchmark_strategies_do_not_run_any_solver(self):
        for strategy in ("exact-only", "hybrid-exact", "hybrid-exact+llm",
                         "quantum+repair+local", "quantum+repair+local+llm"):
            with self.subTest(strategy=strategy), \
                 patch("sys.argv", ["benchmark", "--llm", "--strategies", strategy]), \
                 patch.object(benchmark, "load_suc_data", return_value=(None, [])), \
                 patch.object(benchmark, "solve_benders") as research, \
                 patch.object(benchmark, "solve_full") as reference, \
                 self.assertRaises(SystemExit):
                benchmark.main()
            research.assert_not_called()
            reference.assert_not_called()

    def test_full_cli_reports_bound_separately_from_feasible_objective(self):
        with patch("sys.argv", ["full", "--time-limit", "1"]), \
             patch.object(full, "load_suc_data", return_value=(None, [])), \
             patch.object(full, "solve_full", return_value=reference_result()), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            full.main()
        self.assertIn("status=TIME_LIMIT", output.getvalue())
        self.assertIn("objective=100.000000", output.getvalue())
        self.assertIn("lower_bound=90.000000", output.getvalue())

    def test_reference_changes_do_not_change_research_candidates_or_results(self):
        observations = []
        for reference in (reference_result(600000, 500000), reference_result(650000, 400000)):
            candidates = []

            def record_scenario(data, scenario, u, v, w, **kwargs):
                candidates.append((scenario.name, u.tolist(), v.tolist(), w.tolist()))
                return solve_scenario(data, scenario, u, v, w, **kwargs)

            with patch("sys.argv", ["benchmark", "--data", str(ROOT / "data/suc_10x24.json"),
                                    "--strategies", "quantum+repair+decommit", "--max-iterations", "1"]), \
                 patch.object(benchmark, "_config", return_value=BendersConfig(
                     max_iterations=1, quantum_candidates=False,
                 )), \
                 patch.object(benchmark, "solve_benders", wraps=solve_benders), \
                 patch("suc_benders.solver.solve_scenario", side_effect=record_scenario), \
                 patch.object(benchmark, "solve_full", return_value=reference), \
                 patch.object(benchmark, "print", create=True) as output:
                benchmark.main()
            payload = json.loads(output.call_args.args[0])
            row = copy.deepcopy(payload["rows"][0])
            for key in ("runtime", "relative_error", "reference_gap"):
                row.pop(key)
            observations.append((row, candidates))
        self.assertTrue(observations[0][1])
        self.assertEqual(observations[0], observations[1])


if __name__ == "__main__":
    unittest.main()
