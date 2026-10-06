import math
from pathlib import Path
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from suc_benders.gap_closure_experiment import (PHASE_SECONDS, TOTAL_SECONDS, configuration,
    definitions, end_worker, reference_record, report, select_hours, update_checkpoint)
from suc_benders.sa_scale_experiment import append, save
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.candidates import make_candidate
from suc_benders.block_master_model import solve_master
from tests.test_temporal_solver import tiny, enumerated_run, EnumeratedQubo


class NativeFallbackTests(unittest.TestCase):
    def test_disabled_path_never_requests_classical_point(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_master", side_effect=AssertionError("primal channel disabled")):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1))
        self.assertEqual(result.status, "DUAL_CONVERGED")
        self.assertEqual(result.diagnostics["native_calls"], 0)

    def test_third_stalled_iteration_and_reset_after_attempt(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.certify_master", return_value=dict(lower_bound=0., seconds=0., status=2)), patch(
                "suc_benders.paper_solver.converged", return_value=False), patch(
                "suc_benders.paper_solver.solve_master", wraps=solve_master):
            result = solve_benders(data, scenarios, BendersConfig(max_inner_iterations=7, native_fallback_patience=3))
        self.assertEqual([r["inner"] for r in result.diagnostics["native_candidates"]], [3, 6])
        self.assertEqual(result.status, "INNER_NOT_CONVERGED")
        self.assertTrue(all(not r["accepted"] for r in result.diagnostics["native_candidates"] if r["duplicate"]))

    def test_native_only_uses_real_dispatch_and_certification_without_sa(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_qubo", side_effect=AssertionError("SA disabled")):
            result = solve_benders(data, scenarios, BendersConfig(block_hours=1, quantum_candidates=False,
                                  native_fallback_patience=1))
        self.assertEqual(result.status, "DUAL_CONVERGED", result.message)
        self.assertGreater(result.diagnostics["native_calls"], 0)
        self.assertLessEqual(result.diagnostics["reliable_lower"], result.objective+1e-5)

    def test_deferred_recovery_preserves_bounds_and_runs_after_inner_work(self):
        data, scenarios = tiny()
        emitted = []
        result = solve_benders(data, scenarios, BendersConfig(block_hours=1, quantum_candidates=False,
            native_fallback_patience=1, defer_full_recovery=True, event_sink=emitted.append))
        self.assertEqual(result.status, "DUAL_CONVERGED")
        from suc_benders.validation import validate_schedule
        validate_schedule(data, scenarios, result)
        for start in [r for r in emitted if r['kind'] == 'deferred_recovery_start']:
            previous = [r for r in emitted if r['kind'] == 'inner_iteration' and
                        (r['outer'], r['block']) == (start['outer'], start['block'])]
            self.assertTrue(previous)
            self.assertLessEqual(previous[-1]['clock'], start['clock'])

    def test_missing_native_point_is_recorded_without_manufacturing_progress(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.solve_master", return_value=dict(status=9, seconds=0., point=None)), patch(
                "suc_benders.paper_solver.converged", return_value=False):
            result = solve_benders(data, scenarios, BendersConfig(max_inner_iterations=3, native_fallback_patience=3))
        row = result.diagnostics["native_candidates"][0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection"], "missing_point")
        self.assertEqual(result.status, "INNER_NOT_CONVERGED")

    def test_local_improvement_resets_three_round_stagnation(self):
        original, _ = tiny(1)
        vectors = {k: np.repeat(v, 2) for k, v in vars(original).items()
                   if isinstance(v, np.ndarray) and k != "base_load"}
        vectors.update(cost_a=np.zeros(2), cost_c=np.array([100., 200.]))
        data = replace(original, n_units=2, **vectors)
        from suc_benders.data import Scenario
        scenarios = [Scenario("tiny", 1., data.base_load)]
        baseline = make_candidate(data, scenarios, np.ones((2, 1), dtype=int))
        call = 0
        def sample(*args, **kwargs):
            nonlocal call
            call += 1
            target = 2 if call < 3 else 1
            return [next(p for p in EnumeratedQubo.current if sum(v for k, v in p.items() if k[0] == "u") == target)]
        with patch("suc_benders.paper_solver.CompactQubo", EnumeratedQubo), patch(
                "suc_benders.paper_solver.solve_qubo", side_effect=sample), patch(
                "suc_benders.paper_solver.repair_commitment", return_value=baseline), patch(
                "suc_benders.paper_solver.converged", return_value=False):
            result = solve_benders(data, scenarios, BendersConfig(max_inner_iterations=6, native_fallback_patience=3))
        self.assertEqual([r["inner"] for r in result.diagnostics["native_candidates"]], [6])

    def test_patience_validation_rejects_zero_negative_bool_and_float(self):
        data, scenarios = tiny()
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solve_benders(data, scenarios, BendersConfig(native_fallback_patience=value))

    def test_provenance_survives_price_changes_and_stitching(self):
        data, scenarios = tiny()
        emitted = []
        result = solve_benders(data, scenarios, BendersConfig(block_hours=1, quantum_candidates=False,
            native_fallback_patience=1, event_sink=emitted.append))
        self.assertTrue(all("candidate_origins" in r for r in emitted if r["kind"] == "evaluation"))
        bounds = [r for r in emitted if r["kind"] == "bound_update"]
        self.assertTrue(bounds)
        self.assertTrue(all(math.isfinite(r["reliable_lower"]) for r in bounds))
        self.assertTrue(all(r["clock"] <= emitted[0]["deadline_clock"] for r in bounds))


class GapClosureProtocolTests(unittest.TestCase):
    def test_worker_does_not_read_reference_data(self):
        data, scenarios = tiny(1)
        definition = dict(stage="deterministic", case="tiny", method="native_only", hours=1, seed=7, seconds=30)
        with tempfile.TemporaryDirectory() as temp, patch(
                "suc_benders.gap_closure_experiment.load_case", return_value=(data, scenarios)), patch(
                "suc_benders.gap_closure_experiment.read", side_effect=AssertionError("reference read")):
            end_worker(Path(temp), definition, Path(temp), time.perf_counter()+30)
            self.assertTrue((Path(temp)/"result.json").exists())

    def test_report_rebuild_and_late_invalid_bound_handling(self):
        data, scenarios = tiny(1)
        emitted = []
        result = solve_benders(data, scenarios, BendersConfig(quantum_candidates=False,
                              native_fallback_patience=1, event_sink=emitted.append))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            runs = [r for r in definitions() if r["stage"] == "confirmation"]
            save(directory / "manifest.json", dict(protocol="test", runs=runs, commit="test",
                 source_sha256={}, input_sha256={}, historical_sha256={}))
            save(directory / "selection.json", dict(chosen_hours=1))
            save(directory / "state.json", dict(status="completed", within_total_budget=True))
            save(directory / "references/n32-stoch/result.json", dict(certified=True, objective=result.objective,
                 lower_bound=result.objective, mip_gap=0., validation=dict(feasible_solution=True)))
            save(directory / "references/n32-stoch/state.json", dict(status="completed"))
            deadline = emitted[0]["clock"]-.1+600.
            for run in runs:
                folder = directory / "runs" / run["id"]
                save(folder / "state.json", dict(status="time_limit", deadline=deadline, started_clock=emitted[0]["clock"]-.1))
                with (folder/"events.jsonl").open("w", encoding="utf-8") as stream:
                    for row in emitted:
                        append(stream, row)
                    append(stream, dict(kind="bound_update", clock=deadline+1., reliable_lower=1e99))
            with patch("suc_benders.gap_closure_experiment.load_case", return_value=(data, scenarios)), patch(
                    "suc_benders.gap_closure_experiment.plot"):
                first = report(directory)
                second = report(directory)
                self.assertEqual(first, second)
                self.assertTrue(first["primary_passed"])
                self.assertTrue(all(r["late_events"] == 1 for r in first["runs"]))
                folder = directory / "runs" / runs[0]["id"]
                with (folder/"events.jsonl").open("a", encoding="utf-8") as stream:
                    append(stream, dict(kind="bound_update", clock=deadline-.1, reliable_lower=1e99,
                         coordination_upper=None, outer=0, block=None, inner=0, bound_source="invalid"))
                broken = report(directory)
                self.assertFalse(broken["primary_passed"])
                self.assertIsNone(broken["runs"][0]["paper_gap"])

    def test_fixed_matrix_budget_and_arm_configuration(self):
        runs = definitions()
        self.assertEqual(len(runs), 23)
        self.assertEqual(TOTAL_SECONDS, 7200)
        for stage, seconds in PHASE_SECONDS.items():
            if stage not in ("reference", "audit"):
                self.assertEqual(sum(r["seconds"] for r in runs if r["stage"] == stage), seconds)
        for run in runs:
            cfg = configuration({**run, "hours": run["hours"] or 1})
            self.assertFalse(cfg.llm.enabled)
            self.assertEqual(cfg.quantum.candidate_pool_size, 16)
            self.assertEqual(cfg.defer_full_recovery, not (run['stage']=='sanity' and run['method']=='sa_only'))
            self.assertEqual(cfg.native_fallback_patience,
                             {"sa_only": None, "native_only": 1, "sa_fallback": 3}[run["method"]])

    def test_checkpoint_rejects_late_values_and_incomplete_evaluations(self):
        checkpoint = {}
        late = dict(kind="bound_update", clock=11., reliable_lower=99.)
        self.assertFalse(update_checkpoint(checkpoint, late, 10.))
        self.assertEqual(checkpoint, {})
        self.assertFalse(update_checkpoint(checkpoint, dict(kind="evaluation_incomplete", clock=9.), 10.))
        self.assertTrue(update_checkpoint(checkpoint, dict(kind="bound_update", clock=9.,
                        reliable_lower=90., coordination_upper=None), 10.))
        self.assertEqual(checkpoint["bound"]["reliable_lower"], 90.)

    def test_selection_uses_certified_upper_and_never_treats_timeout_as_ceiling(self):
        def run(hours, lower, upper=None, valid=True):
            return dict(hours=hours, reliable_lower=lower, coordination_upper=upper, numerical_audit_passed=valid)
        selected = select_hours([run(1, 95.), run(2, 95.+1e-6), run(3, 97., 98.), run(5, 99., valid=False)], 100.)
        self.assertEqual(selected["chosen_hours"], 1)
        self.assertEqual([r["reason"] for r in selected["excluded"]], ["certified_dual_ceiling", "numerical_audit"])

    def test_reference_requires_complete_on_time_certification_and_physical_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            save(d / "references/n32-stoch/result.json", dict(certified=True, objective=100., lower_bound=100.,
                 mip_gap=0., clock=11., validation=dict(feasible_solution=True)))
            save(d / "references/n32-stoch/state.json", dict(status="completed", deadline=10.))
            self.assertFalse(reference_record(d, "n32-stoch")["certified"])
            self.assertFalse(reference_record(d, "n10-det")["certified"])


if __name__ == "__main__":
    unittest.main()
