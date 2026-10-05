import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from suc_benders.sa_scale_experiment import (SUBSETS, case_input, completed_in_time, config,
    definitions, frozen_sampler_input, seed_for, trial_statistics, events, supervise, read, save, screening_verdict)


class ScaleProtocolTests(unittest.TestCase):
    def test_nested_subsets_and_capacity_proportional_load(self):
        full = case_input(32)
        previous = set()
        for n, capacity in ((3, 331), (5, 548), (10, 1515), (32, 3405)):
            data = case_input(n)
            self.assertTrue(previous.issubset(SUBSETS[n]))
            previous = set(SUBSETS[n])
            self.assertEqual(data["n_units"], n)
            self.assertEqual(data["horizon"], 24)
            self.assertEqual(sum(data["pmax"]), capacity)
            np.testing.assert_allclose(data["base_load"], np.array(full["base_load"])*capacity/3405)
            for k, values in full.items():
                if isinstance(values, list) and k != "base_load":
                    self.assertEqual(data[k], [values[i-1] for i in SUBSETS[n]])

    def test_prespecified_counts_pairs_and_peak_blocks(self):
        d = definitions()
        self.assertEqual([len(d[k]) for k in ("snapshots", "trials", "runs")], [24, 144, 18])
        for s in d["snapshots"]:
            if s["role"] == "peak":
                self.assertIn(11, range(s["block"]*s["hours"], (s["block"]+1)*s["hours"]))
            pairs = [t for t in d["trials"] if t["snapshot"] == s["id"]]
            self.assertEqual(len(pairs), 6)
            self.assertEqual({t["mode"] for t in pairs}, {"fixed", "varying"})
        for field in ("snapshots", "trials", "runs"):
            self.assertEqual(len({v["id"] for v in d[field]}), len(d[field]))

    def test_seeds_and_nonproduction_configuration(self):
        self.assertEqual([seed_for(7, "fixed", i) for i in range(3)], [7, 7, 7])
        self.assertEqual([seed_for(7, "varying", i) for i in range(3)], [7, 8, 9])
        self.assertEqual(config().max_inner_iterations, 60)
        from suc_benders.solver import BendersConfig
        self.assertEqual(BendersConfig().max_inner_iterations, 20)
        self.assertFalse(config().llm.enabled)

    def test_sampling_input_has_no_reference_or_witness_access(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez_compressed(folder / "qubo.npz", matrix=np.eye(2))
            (folder / "sampling.json").write_text(json.dumps({"energy_unit": .1}))
            for name in ("snapshot.json", "reference.json", "frozen.json"):
                (folder / name).write_text("invalid poisoned oracle/witness contents")
            matrix, metadata = frozen_sampler_input(folder)
            np.testing.assert_array_equal(matrix, np.eye(2))
            self.assertEqual(metadata, {"energy_unit": .1})

    def test_timeout_does_not_create_completed_batch(self):
        self.assertTrue(completed_in_time(5., 5.))
        self.assertFalse(completed_in_time(5.0001, 5.))
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with patch("suc_benders.sa_scale_experiment.subprocess.Popen") as launch:
                supervise(p, p / "trial", "sample", "ignored", 5, 0)
                launch.assert_not_called()
            state = json.loads((p / "trial/state.json").read_text())
            self.assertEqual(state["status"], "phase_budget_exhausted")
            (p / "partial.jsonl").write_text('{"call":0}\n{"call":')
            self.assertEqual(events(p / "partial.jsonl"), [{"call": 0}])

    def test_transient_windows_replace_race_retries_without_losing_deadline(self):
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text",
                side_effect=[PermissionError(), PermissionError(), '{"deadline":5}']), patch(
                "suc_benders.sa_scale_experiment.time.sleep") as pause:
            self.assertEqual(read(Path("active.json")), {"deadline": 5})
            self.assertEqual(pause.call_count, 2)

    def test_transient_windows_write_race_retries_the_same_payload(self):
        with tempfile.TemporaryDirectory() as temp, patch(
                "suc_benders.sa_scale_experiment.publish", side_effect=[PermissionError(), None]) as writer, patch(
                "suc_benders.sa_scale_experiment.time.sleep"):
            save(Path(temp)/"active.json", {"deadline": 5})
            self.assertEqual(writer.call_count, 2)
            self.assertEqual(writer.call_args_list[0], writer.call_args_list[1])

    def test_failed_or_censored_trial_cannot_pass_a_three_seed_screen(self):
        good = {"reference_certified": True, "samples": 16, "relative_gap": .001,
                "state": {"status": "completed"}}
        self.assertEqual(screening_verdict([good]*3), "passed")
        self.assertEqual(screening_verdict([good, good, {**good, "state": {"status": "failed"}}]), "inconclusive")
        self.assertEqual(screening_verdict([good, good, {**good, "relative_gap": None}]), "not_passed")

    def test_metric_denominators_and_missing_feasibility(self):
        reference = {"certified": True, "objective": -10., "native_checks": {"feasible": True}}
        def sample(value, discrete, native, encoded):
            return {"raw_objective": value, "raw_uvw": [int(discrete)],
                    "native": {"feasible": native, "discrete": {"feasible": discrete}},
                    "encoded": {"feasible": encoded}}
        samples = [sample(-9., True, True, False), sample(None, False, False, False)]
        batches = [{"samples": [[1], [0]], "seconds": .2}]*2
        stats = trial_statistics(samples, batches, reference)
        self.assertEqual(stats["native_feasible_rate"], .5)
        self.assertEqual(stats["encoded_feasible_rate"], 0.)
        self.assertAlmostEqual(stats["relative_gap"], .1)
        self.assertEqual(stats["repeated_batch_fraction"], .5)
        empty = trial_statistics([], [], reference)
        self.assertIsNone(empty["relative_gap"])
        self.assertIsNone(empty["native_feasible_rate"])
        missing = trial_statistics(samples, batches, {})
        self.assertIsNone(missing["relative_gap"])


if __name__ == "__main__":
    unittest.main()
