import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from suc_benders.data import DataError, load_suc_data
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.subproblem import solve_scenario
from suc_benders import rts24_experiment as experiment
from suc_benders.attribution import write_json
from suc_benders.data import Scenario
from tests.test_capacity import one_hour_data

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/rts24"
spec = importlib.util.spec_from_file_location("rts24_convert", CASE / "convert.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class RTS24DataTests(unittest.TestCase):
    def test_pinned_conversion_and_mapping(self):
        generated = converter.convert()
        for name, body in generated.items():
            self.assertEqual(json.loads((CASE / name).read_text(encoding="utf-8")), body)
        data = generated["units.json"]
        self.assertEqual((data["n_units"], data["horizon"]), (32, 24))
        self.assertEqual(sum(data["pmax"]), 3405)
        self.assertEqual(max(data["base_load"]), 2850)
        self.assertEqual(data["base_load"][14], 2850)
        self.assertAlmostEqual(sum(data["initial_output"]), 3023.3)
        self.assertEqual(data["ramp_up"][0], 180)
        self.assertEqual(data["pstart"][0], 20)
        self.assertEqual(data["cost_a"][2], .014142)
        self.assertEqual(data["startup_cost"], [1500] * 32)
        mapping = generated["provenance.json"]["generator_mapping"]
        self.assertEqual([r["source_generator_id"] for r in mapping], list(range(1, 15)) + list(range(16, 34)))
        self.assertEqual(mapping[14]["matpower_bus"], 15)

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in converter.SOURCES:
                (path / name).write_bytes((CASE / "source" / name).read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                converter.convert(path)

    def test_profiles(self):
        data, base = load_suc_data(CASE / "units.json", CASE / "deterministic.json")
        self.assertEqual(data.reserve_fraction, 0)
        self.assertEqual(len(base), 1)
        data, stochastic = load_suc_data(CASE / "units.json", CASE / "stochastic.json")
        self.assertEqual(data.reserve_fraction, .05)
        self.assertEqual([s.probability for s in stochastic], [.25, .5, .25])
        self.assertAlmostEqual(max(stochastic[2].load) * 1.05, 3142.125)
        self.assertEqual(data.initial_output[2], 76)
        self.assertNotEqual(data.initial_output[2], data.pmin[2])

    def test_explicit_initial_output_validation_and_legacy_default(self):
        obj = json.loads((ROOT / "data/suc_10x24.json").read_text(encoding="utf-8"))
        legacy, _ = load_suc_data(ROOT / "data/suc_10x24.json")
        np.testing.assert_array_equal(legacy.initial_output, legacy.pmin)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "units.json"
            for invalid in ([], [float("nan")] * 10, [float("inf")] * 10,
                            [0] * 10, [9999] * 10):
                path.write_text(json.dumps({**obj, "initial_output": invalid}), encoding="utf-8")
                with self.assertRaises(DataError):
                    load_suc_data(path)
            obj["initial_duration"][0] = -8
            initial = obj["pmin"].copy()
            path.write_text(json.dumps({**obj, "initial_output": initial}), encoding="utf-8")
            with self.assertRaisesRegex(DataError, "offline"):
                load_suc_data(path)
            initial[0] = 0
            path.write_text(json.dumps({**obj, "initial_output": initial}), encoding="utf-8")
            actual, _ = load_suc_data(path)
            np.testing.assert_array_equal(actual.initial_output, initial)

    def test_initial_output_affects_first_hour_in_both_models(self):
        obj = dict(n_units=1, horizon=1, pmin=[10], pmax=[100], pstart=[100], pshut=[100],
                   min_up=[1], min_down=[1], ramp_up=[5], ramp_down=[5], initial_duration=[1],
                   base_load=[80], cost_a=[.01], cost_b=[2], cost_c=[3], startup_cost=[0], shutdown_cost=[0])
        with tempfile.TemporaryDirectory() as directory:
            path, scenario_path = Path(directory) / "units.json", Path(directory) / "scenarios.json"
            scenario_path.write_text(json.dumps({"reserve_fraction": 0, "scenarios": [
                {"name": "base", "probability": 1, "load_factor": 1}]}), encoding="utf-8")
            for explicit in (False, True):
                path.write_text(json.dumps({**obj, **({"initial_output": [80]} if explicit else {})}), encoding="utf-8")
                data, scenarios = load_suc_data(path, scenario_path)
                full = solve_full(data, scenarios, FullConfig(output_flag=0))
                sub = solve_scenario(data, scenarios[0], np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
                self.assertEqual(sub.feasible, explicit)
                self.assertEqual(np.isfinite(full.objective), explicit)
                if explicit:
                    self.assertAlmostEqual(full.objective, sub.objective + 3, places=5)


class RTS24ProtocolTests(unittest.TestCase):
    def test_fixed_protocol(self):
        config = experiment.configuration()
        self.assertEqual((config.max_iterations, config.time_limit_seconds), (3, 600))
        self.assertEqual((config.quantum.kaiwu_seed, config.quantum.candidate_pool_size), (7, 16))
        self.assertEqual((config.master_encoding, config.decomposition), ("dynamic", "none"))
        self.assertFalse(config.llm.enabled or config.decommit_candidates or config.random_candidates)

    def test_existing_directory_never_starts_another_run(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.argv", ["rts24", "--output-dir", directory]), \
                patch.object(experiment, "supervise") as supervise:
            with self.assertRaisesRegex(ValueError, "must be new"):
                experiment.main()
            supervise.assert_not_called()

    def test_audit_rejects_late_candidates_and_corrupt_dispatch(self):
        data = one_hour_data()
        scenarios = [Scenario("base", 1, data.base_load)]
        row = dict(objective=81., u=[[1, 0, 0]], v=[[1, 0, 0]], w=[[0, 1, 0]],
                   scenarios=[dict(name="base", objective=69., p=[[30., 0, 0]], r=[[0., 0, 0]])])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiment, "load_suc_data", return_value=(data, scenarios)):
            root = Path(directory)
            write_json(root / "manifest.json", {"source_sha256": {}, "commit": "test", "protocol": {}})
            for profile in experiment.PROFILES:
                sa, ref = root / profile / "sa", root / profile / "reference"
                write_json(sa / "state.json", {"status": "completed"})
                write_json(ref / "state.json", {"status": "completed"})
                write_json(sa / "result.json", {**row, "diagnostics": {"proposals": [{"origin": "sa"}]}})
                write_json(ref / "result.json", {**row, "status": "OPTIMAL", "mip_gap": 0, "lower_bound": 80})
                write_json(ref / "fixed_dispatch_check.json", {"all_feasible": True, "difference": 0, "tolerance": .001,
                    "result_sha256": experiment.hash_file(ref / "result.json")})
                events = [dict(kind="sa_end", status="completed", elapsed_seconds=1),
                          dict(kind="iteration", elapsed_seconds=3),
                          dict(kind="evaluation", feasible=True, elapsed_seconds=2, **row)]
                (sa / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            report = experiment.audit(root)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["profiles"]["deterministic"]["certified_relative_error_bound"], 1/80)
            path = root / "deterministic/sa/events.jsonl"
            events[-1]["elapsed_seconds"] = 601
            path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            self.assertFalse(experiment.audit(root)["passed"])
            events[-1]["elapsed_seconds"] = 2
            events[-1]["scenarios"][0]["p"][0][0] = 31
            path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            with self.assertRaises(AssertionError):
                experiment.audit(root)


if __name__ == "__main__":
    unittest.main()
