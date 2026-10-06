import math
import time
import unittest
import itertools
from unittest.mock import patch

from suc_benders.block_master_model import block_dual_floor, continuous_relaxation, priced_blocks
from suc_benders.paper_model import Cut, IndexedUC, Row, key
from suc_benders.pure_sa import active_cuts, interval_fixed_binary
from suc_benders.pure_sa_gap_experiment import PHASE_SECONDS, TOTAL_SECONDS, configuration, definitions
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.validation import validate_schedule
from tests.test_temporal_solver import EnumeratedQubo, enumerate_points, enumerated_run, tiny


class PureSAIsolationTests(unittest.TestCase):
    def test_pure_search_never_calls_integer_oracles(self):
        data, scenarios = tiny(1)
        emitted = []
        with patch("suc_benders.paper_solver.solve_master", side_effect=AssertionError("integer candidate")), patch(
                "suc_benders.paper_solver.certify_master", side_effect=AssertionError("integer certificate")):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, max_inner_iterations=2,
                gap_experiment_variant="sa_strict", defer_full_recovery=True, event_sink=emitted.append))
        validate_schedule(data, scenarios, result)
        self.assertEqual(result.diagnostics["online_integer_master_calls"], 0)
        self.assertTrue(any(r["kind"] == "multiplier_epoch" for r in emitted))
        self.assertTrue(any(r["kind"] == "local_cut" for r in emitted))
        self.assertLessEqual(result.diagnostics["reliable_lower"], result.objective+1e-5)

    def test_pure_variant_rejects_native_candidate_configuration(self):
        data, scenarios = tiny(1)
        for variant in ("sa_strict", "sa_flow", "sa_compact", "sa_full"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                solve_benders(data, scenarios, BendersConfig(gap_experiment_variant=variant,
                              native_fallback_patience=3))
        with self.assertRaises(ValueError):
            solve_benders(data, scenarios, BendersConfig(gap_experiment_variant="unknown"))

    def test_fixed_matrix_and_budget(self):
        self.assertEqual(TOTAL_SECONDS, 14400)
        runs = definitions()
        self.assertEqual(len(runs), 23)
        for stage, seconds in PHASE_SECONDS.items():
            if stage not in ("reference", "audit"):
                self.assertEqual(sum(r["seconds"]+r["audit_seconds"] for r in runs if r["stage"] == stage), seconds)
        for run in runs:
            if run["method"] == "selected":
                with self.assertRaises(ValueError):
                    configuration(run)
                continue
            cfg = configuration({**run, "hours": run["hours"] or 1})
            native = run["method"] == "native_diagnostic"
            self.assertEqual(cfg.quantum_candidates, not native)
            self.assertEqual(cfg.native_fallback_patience, 1 if native else None)
            self.assertEqual(cfg.quantum.candidate_pool_size, 16)
            self.assertEqual(cfg.max_inner_iterations, 200 if run["method"] == "sa_strict" else 8)
            self.assertFalse(cfg.llm.enabled)
        final = [r for r in runs if r["stage"] == "confirmation"]
        self.assertEqual([r["seed"] for r in final], [7, 100007, 200007])
        self.assertTrue(all(r["seconds"] == 1200 and r["audit_seconds"] == 600 for r in final))


class PureSAFlowTests(unittest.TestCase):
    def test_analytic_dual_floors_under_price_changes_by_enumeration(self):
        data, scenarios = tiny(3)
        uc = IndexedUC(data, scenarios)
        relaxation = continuous_relaxation(uc, time.perf_counter()+30)
        initial = relaxation["multipliers"]
        blocks, cross = priced_blocks(uc, 1, initial)
        self.assertGreaterEqual(sum(block_dual_floor(b, initial) for b in blocks),
                                relaxation["lower_bound"]-1e-3)
        changed = list(initial)
        for i in cross:
            changed[i] = -3. if uc.rows[i].equality else 2.
        for prices in (initial, changed):
            blocks, _ = priced_blocks(uc, 1, prices)
            for block in blocks:
                candidates = [block.dispatch(x, block.costs, time.perf_counter()+30)
                              for x in enumerate_points(block)]
                optimum = min(r["cost"]+block.constant for r in candidates if r and r["feasible"])
                floor = block_dual_floor(block, initial)
                self.assertTrue(math.isfinite(floor))
                self.assertLessEqual(floor, optimum+1e-5)

    def test_inexact_round_visits_all_blocks_without_false_inner_convergence(self):
        data, scenarios = tiny(3)
        emitted = []
        with patch("suc_benders.paper_solver.CompactQubo", EnumeratedQubo), patch(
                "suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.converged", return_value=False), patch(
                "suc_benders.paper_solver.certify_master", side_effect=AssertionError("integer certification")):
            result = solve_benders(data, scenarios, BendersConfig(block_hours=1, max_iterations=2,
                max_inner_iterations=200, gap_experiment_variant="sa_flow", event_sink=emitted.append))
        validate_schedule(data, scenarios, result)
        self.assertEqual(result.diagnostics["sa_calls"], 2*3*8)
        self.assertEqual(len(result.diagnostics["blocks"]), 6)
        self.assertTrue(all(not b["converged"] for b in result.diagnostics["blocks"]))
        self.assertEqual(len([r for r in emitted if r["kind"] == "multiplier_update"]), 2)
        approximate = [r for r in emitted if r["kind"] == "approximate_outer"]
        self.assertEqual(len(approximate), 2)
        self.assertTrue(all(not r["all_inner_converged"] for r in approximate))
        for outer in (1, 2):
            self.assertLessEqual(sum(r["iteration"] == outer for r in
                                     emitted if r["kind"] == "evaluation"), 2)
        self.assertTrue(all("candidate_origins" in r for r in emitted if r["kind"] == "evaluation"))
        self.assertEqual(result.diagnostics["online_integer_master_calls"], 0)


class PureSACompressionTests(unittest.TestCase):
    def test_long_row_interval_propagation_preserves_all_binary_solutions(self):
        keys = [key("u", i, 0) for i in range(4)]
        rows = [Row({k: -1. for k in keys}, 4., False, 1., "long_hard")]
        fixed = interval_fixed_binary(rows, {})
        self.assertEqual(fixed, dict.fromkeys(keys, 1))
        for bits in itertools.product((0, 1), repeat=4):
            point = dict(zip(keys, bits))
            if rows[0].value(point) <= 0:
                self.assertTrue(all(point[k] == v for k, v in fixed.items()))
        equality = Row({k: 1. for k in keys}, -1., True, 1., "sum_one")
        fixed = interval_fixed_binary([equality], {keys[0]: 1})
        self.assertEqual(fixed, {keys[0]: 1, **dict.fromkeys(keys[1:], 0)})
        with self.assertRaises(ValueError):
            interval_fixed_binary(rows, {keys[0]: 0})

    def test_active_cuts_deduplicate_only_for_encoding_and_include_violations(self):
        k = key("u", 0, 0)
        cuts = [Cut({k: float(i)}, float(-i), True) for i in range(25)]
        cuts += [Cut({k: 1.}, -10., False), Cut({k: 1.}, -.5, False)]
        original = list(cuts)
        active = active_cuts(cuts, {k: 1}, [({k: 1}, -20.)])
        self.assertLessEqual(len(active), 16)
        self.assertEqual(cuts, original)
        self.assertIn(cuts[-1], active)
        self.assertNotIn(cuts[-2], active)
        self.assertIn(cuts[-3], active)

    def test_compact_search_keeps_complete_unquantized_cut_records(self):
        data, scenarios = tiny(3)
        emitted = []
        result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, max_iterations=2,
            gap_experiment_variant="sa_compact", event_sink=emitted.append))
        validate_schedule(data, scenarios, result)
        self.assertTrue(all(r["active_cuts"] <= 16 for r in emitted if r["kind"] == "sampling"))
        cuts = [r for r in emitted if r["kind"] == "local_cut"]
        self.assertTrue(cuts)
        self.assertTrue(all(set(r["cut"]) == {"a", "c", "feasible"} for r in cuts))
        self.assertEqual(result.diagnostics["online_integer_master_calls"], 0)


if __name__ == "__main__":
    unittest.main()
