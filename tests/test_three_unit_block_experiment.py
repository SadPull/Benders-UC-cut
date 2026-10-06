import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from suc_benders.sa_scale_experiment import case_input, config, read, save, worker
from suc_benders.three_unit_block_experiment import (CASE, HOURS, RUN_SECONDS,
    TOTAL_SECONDS, convergence_trace, definitions, report)


class ThreeUnitBlockTests(unittest.TestCase):
    def test_protocol_is_nine_fixed_runs_with_unchanged_search_settings(self):
        runs = definitions()
        self.assertEqual([(r["hours"], r["seed"]) for r in runs],
                         [(h, seed) for h in (1, 3, 5) for seed in (7, 100007, 200007)])
        self.assertEqual(len({r["id"] for r in runs}), 9)
        self.assertEqual(sum(case_input(3)["pmax"]), 331)
        self.assertEqual(case_input(3)["horizon"], 24)
        self.assertEqual((RUN_SECONDS, TOTAL_SECONDS), (90, 1200))
        for r in runs:
            cfg = config(r["seed"], r["mode"], RUN_SECONDS, r["hours"])
            self.assertEqual((cfg.max_iterations, cfg.max_inner_iterations), (20, 60))
            self.assertTrue(cfg.sa_seed_per_call)
            self.assertFalse(cfg.llm.enabled)
            self.assertEqual(cfg.quantum.candidate_pool_size, 16)

    def test_five_hour_cli_option_is_accepted(self):
        from suc_benders.astra_run import main
        with patch("sys.argv", ["suc_benders", "--block-hours", "5", "--help"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stop:
                main()
        self.assertEqual(stop.exception.code, 0)

    def test_worker_uses_manifest_five_hour_run_without_reference_input(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            definition = next(r for r in definitions() if r["hours"] == 5)
            save(directory / "manifest.json", dict(runs=[definition]))
            args = SimpleNamespace(output_dir=directory, identifier=definition["id"], worker="end", deadline=123.)
            with patch("suc_benders.sa_scale_experiment.end_worker") as solve, patch(
                    "suc_benders.sa_scale_experiment.os._exit"), patch(
                    "suc_benders.sa_scale_experiment.definitions", side_effect=AssertionError("hardcoded protocol")):
                worker(args)
            solve.assert_called_once_with(directory, definition, directory / "end_to_end" / definition["id"], 123.)

    def test_trace_separates_blocks_and_prices_and_only_counts_completed_outer_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            definition = definitions()[0]
            folder = directory / "end_to_end" / definition["id"]
            folder.mkdir(parents=True)
            rows = [dict(kind="inner_iteration", outer=outer, block=block, inner=1,
                         lower=lower, upper=lower+2, cuts=1, converged=False)
                    for outer, block, lower in [(1, 0, 10.), (1, 1, -30.), (2, 0, -100.)]]
            rows.append(dict(kind="iteration", iteration=1, elapsed_seconds=2., incumbent=200.,
                             reliable_lower=150., coordination_upper=200.))
            (folder / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
            trace = convergence_trace(directory, definition)
            self.assertEqual([(r["outer"], r["block"]) for r in trace["blocks"]], [(1, 0), (1, 1), (2, 0)])
            self.assertEqual(len(trace["outer"]), 1)
            self.assertEqual([r["gap"] for r in trace["inner"]], [2., 2., 2.])

    def test_report_preserves_timeout_and_missing_reference_without_fake_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            save(directory / "inputs" / CASE / "units.json", case_input(3))
            save(directory / "inputs" / CASE / "scenarios.json", dict(reserve_fraction=0.,
                 scenarios=[dict(name="base", probability=1., load_factor=1.)]))
            save(directory / "manifest.json", dict(runs=definitions(), source_sha256={}, input_sha256={},
                commit="test", dirty_paths=[], block_lengths={str(h): [min(h, 24-t) for t in range(0, 24, h)] for h in HOURS}))
            for r in definitions():
                save(directory / "end_to_end" / r["id"] / "state.json", dict(status="time_limit", wall_seconds=90.))
            with patch("suc_benders.three_unit_block_experiment.plot_traces"):
                summary = report(directory)
            self.assertFalse(summary["baseline_reproduced"])
            self.assertTrue(read(directory / "audit.json")["all_nine_slots_recorded"])
            self.assertEqual(read(directory / "audit.json")["full_results_saved"], 0)
            for r in summary["end_to_end"]:
                self.assertFalse(r["dual_converged"])
                self.assertFalse(r["reached_reference_optimum"])
                self.assertIsNone(r["primal_gap"])
                self.assertIsNone(r["paper_style_gap"])
                self.assertEqual(r["completed_outer_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
