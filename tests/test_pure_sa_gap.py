import math
import time
import unittest
import itertools
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

from suc_benders.block_master_model import block_dual_floor, continuous_relaxation, priced_blocks
from suc_benders.paper_model import Cut, IndexedUC, Row, key
from suc_benders.pure_sa import active_cuts, diverse_candidates, interval_fixed_binary
from suc_benders.pure_sa_gap_experiment import (PHASE_SECONDS, TOTAL_SECONDS, certification_worker,
    configuration, definitions, report, search_hashes, search_worker, select_hours, select_variant, statistics)
from suc_benders.pure_sa_certification import certify_views, coalesce_epochs, committed_events, snapshot
from suc_benders.sa_scale_experiment import append, read, save
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


class PureSADiversityTests(unittest.TestCase):
    def test_selection_checks_full_cuts_deduplicates_and_prefers_diversity(self):
        keys = [key("u", i, 0) for i in range(40)]
        base = dict.fromkeys(keys, 0)
        near = {**base, keys[0]: 1}
        diverse = {**base, keys[1]: 1, keys[2]: 1}
        prohibited = {**base, keys[3]: 1}
        block = SimpleNamespace(bkeys=keys, fixed={}, constant=0., costs=dict.fromkeys(keys, 1.),
            discrete_checks=lambda p: {"feasible": True}, objective_bounds=lambda *a: (0., 100.))
        cuts = [Cut({keys[3]: 1.}, -.5, False)]
        samples = [(base, 0.), (near, 0.), (diverse, 0.), (prohibited, 0.), (base, 0.)]
        indices, reasons = diverse_candidates(block, cuts, samples, {}, limit=2)
        self.assertEqual(indices, {0, 2})
        self.assertEqual(reasons[3], "known_feasibility_cut")
        self.assertEqual(reasons[4], "duplicate")
        self.assertEqual(reasons[1], "batch_limit")
        indices, reasons = diverse_candidates(block, cuts, samples, {tuple(base[k] for k in keys)}, limit=4)
        self.assertNotIn(0, indices)
        self.assertEqual(reasons[0], "duplicate")
        self.assertLessEqual(len(indices), 4)

    def test_full_variant_preserves_purity_and_records_rejected_duplicates(self):
        data, scenarios = tiny(3)
        emitted = []
        with patch("suc_benders.paper_solver.certify_master", side_effect=AssertionError("integer certificate")):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, max_iterations=2,
                gap_experiment_variant="sa_full", event_sink=emitted.append))
        validate_schedule(data, scenarios, result)
        self.assertEqual(result.diagnostics["online_integer_master_calls"], 0)
        self.assertTrue(all(r["dispatch_candidates"] <= 4 for r in emitted if r["kind"] == "sampling"))
        self.assertTrue(any(r.get("rejection") == "duplicate" for r in emitted if r["kind"] == "proposal"))
        for outer in range(1, 3):
            self.assertLessEqual(sum(r["iteration"] == outer for r in emitted if r["kind"] == "evaluation"), 2)


class PureSAProtocolTests(unittest.TestCase):
    def test_offline_unions_only_identical_effective_prices(self):
        data, scenarios = tiny(3)
        indexed = IndexedUC(data, scenarios)
        prices = [0.]*len(indexed.rows)
        blocks, cross = priced_blocks(indexed, 1, prices)
        epoch = dict(outer=1, multipliers=prices, source_clock=1., blocks=[dict(block=j,
            times=list(b.times), constant=b.constant, lower=0., best=None,
            cuts=[dict(a=[], c=1., feasible=True)]) for j,b in enumerate(blocks)])
        same = copy.deepcopy(epoch)
        same.update(outer=2, source_clock=2.)
        local_row = next(i for i in range(len(indexed.rows)) if i not in cross)
        same["multipliers"][local_row] = 2.
        different = copy.deepcopy(epoch)
        different.update(outer=3, source_clock=3.)
        different["multipliers"][cross[0]] = 2.
        grouped = coalesce_epochs(indexed, 1, [epoch, same, different])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["source_epochs"], [1, 2])
        self.assertEqual(len(grouped[0]["blocks"][0]["cuts"]), 2)
        self.assertEqual(grouped[0]["source_clock"], 2.)
        self.assertEqual(len(epoch["blocks"][0]["cuts"]), 1)

    def test_committed_snapshot_ignores_late_and_uncommitted_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"events.jsonl"
            rows = [dict(kind="initial_dual_prices", multipliers=[0.], clock=1., sequence=1),
                dict(kind="multiplier_epoch", outer=1, multipliers=[0.], clock=2., sequence=2,
                     blocks=[dict(block=0, times=[0], constant=0., lower=0.)]),
                dict(kind="local_cut", outer=1, block=0, clock=3., sequence=3, cut=dict(a=[], c=1., feasible=True),
                     feasible=True, cost=1., point=[], sources=["sa"]),
                dict(kind="local_cut", outer=1, block=0, clock=9., sequence=4, cut=dict(a=[], c=1e99, feasible=True),
                     feasible=True, cost=1e99, point=[], sources=["sa"])]
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    append(stream, row)
                    append(stream, dict(kind="event_ready", sequence=row["sequence"], clock=row["clock"]+2.))
                append(stream, dict(kind="local_cut", outer=1, block=0, clock=6., sequence=5))
            committed = committed_events(path)
            self.assertEqual(len(committed), 4)
            view = snapshot(committed, 10.)
            self.assertEqual(len(view["epochs"][0]["blocks"][0]["cuts"]), 1)
            self.assertEqual(view["epochs"][0]["blocks"][0]["best"]["cost"], 1.)

    def test_offline_rebuild_matches_report_and_never_reads_reference_in_search(self):
        data, scenarios = tiny(1)
        final = [r for r in definitions() if r["stage"] == "confirmation"]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            save(directory / "manifest.json", dict(protocol="test", commit="test", runs=final,
                 source_sha256={}, input_sha256={}, historical_sha256={}, reference_sha256={}))
            save(directory / "selection.json", dict(chosen_hours=1, chosen_variant="sa_full"))
            save(directory / "state.json", dict(status="completed", within_total_budget=True))
            for registered in final:
                output = directory / "runs" / registered["id"]
                definition = {**registered, "method":"sa_full", "hours":1}
                save(output / "definition.json", definition)
                start = time.perf_counter()
                save(output / "state.json", dict(status="completed", started_clock=start,
                     deadline=start+1200, allocated_seconds=1200., wall_seconds=1.))
                with patch("suc_benders.pure_sa_gap_experiment.load_case", return_value=(data, scenarios)), patch(
                        "suc_benders.pure_sa_gap_experiment.read", side_effect=AssertionError("search read")), patch(
                        "suc_benders.paper_solver.CompactQubo", EnumeratedQubo), patch(
                        "suc_benders.paper_solver.solve_qubo", side_effect=lambda *a, **k: EnumeratedQubo.current), patch(
                        "suc_benders.paper_solver.certify_master", side_effect=AssertionError("online integer")):
                    search_worker(directory, output, definition, start+1200)
                save(output / "search-hashes.json", search_hashes(output))
                save(output / "offline/state.json", dict(status="completed", deadline=time.perf_counter()+600.,
                     allocated_seconds=600., wall_seconds=1.))
                with patch("suc_benders.pure_sa_gap_experiment.load_case", return_value=(data, scenarios)):
                    certification_worker(directory, output, definition, time.perf_counter()+30.)
            trace = committed_events(output / "events.jsonl")
            best = min((r for r in trace if r["kind"] == "evaluation" and r["feasible"]), key=lambda r:r["objective"])
            save(directory / "references/n32-stoch/result.json", dict(**best, certified=True, mip_gap=0.,
                 lower_bound=best["objective"], validation=dict(feasible_solution=True)))
            save(directory / "references/n32-stoch/state.json", dict(status="completed", deadline=time.perf_counter()+600.))
            with patch("suc_benders.pure_sa_gap_experiment.load_case", return_value=(data, scenarios)), patch(
                    "suc_benders.pure_sa_gap_experiment.plot"):
                first = report(directory)
                self.assertEqual(first, report(directory))
                self.assertTrue(first["primary_passed"])
                self.assertTrue(first["milestone600_passed"])
                self.assertTrue(all(r["online_integer_master_calls"] == 0 for r in first["runs"]))
                audit = read(output / "offline/result.json")
                original_audit = copy.deepcopy(audit)
                audit["checkpoints"]["final"]["lower"] = 1e99
                save(output / "offline/result.json", audit)
                invalid = report(directory)
                self.assertFalse(invalid["primary_passed"])
                self.assertIn("certified_lower_exceeds_feasible_or_dual_upper", invalid["runs"][-1]["errors"])
                save(directory / "references/n32-stoch/state.json", dict(status="failed"))
                self.assertFalse(report(directory)["primary_passed"])
                save(output / "offline/result.json", original_audit)
                save(directory / "references/n32-stoch/state.json", dict(status="completed", deadline=time.perf_counter()+600.))
                bad = copy.deepcopy(best)
                bad.update(sequence=999999, clock=time.perf_counter())
                bad["u"][0][0] = 0
                with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    append(stream, bad)
                    append(stream, dict(kind="event_ready", sequence=999999, clock=time.perf_counter()))
                save(output / "search-hashes.json", search_hashes(output))
                physical = report(directory)
                self.assertFalse(physical["primary_passed"])
                self.assertFalse(physical["runs"][-1]["numerical_audit_passed"])
                self.assertEqual(physical["runs"][-1]["validated_schedules"], first["runs"][-1]["validated_schedules"])

    def test_missing_integer_bound_retains_only_valid_continuous_floors(self):
        data, scenarios = tiny(1)
        emitted = []
        result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, gap_experiment_variant="sa_full",
                                event_sink=emitted.append))
        with patch("suc_benders.pure_sa_certification.solve_master", return_value=dict(status=9, lower_bound=None, seconds=0.)):
            audit = certify_views(data, scenarios, dict(method="sa_full", hours=1), {"final":snapshot(emitted, math.inf)},
                                  time.perf_counter()+30, lambda r: None)
        self.assertFalse(audit["checkpoints"]["final"]["complete"])
        self.assertEqual(audit["checkpoints"]["final"]["certified_models"], 0)
        self.assertLessEqual(audit["checkpoints"]["final"]["lower"], result.objective+1e-5)
        self.assertTrue(any(r.get("reason") == "missing_finite_ObjBound" for r in audit["records"]))

    def test_late_integer_bound_is_discarded(self):
        data, scenarios = tiny(1)
        emitted = []
        result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, gap_experiment_variant="sa_full",
                                event_sink=emitted.append))
        view = snapshot(emitted, math.inf)
        def late(*args, **kwargs):
            time.sleep(.025)
            return dict(status=9, lower_bound=1e99, seconds=.025)
        with patch("suc_benders.pure_sa_certification.solve_master", side_effect=late):
            audit = certify_views(data, scenarios, dict(method="sa_full", hours=1), {"final":view},
                                  time.perf_counter()+.015, lambda r: None)
        self.assertLessEqual(audit["checkpoints"]["final"]["lower"], result.objective+1e-5)
        self.assertTrue(any(r.get("reason") == "late_certificate_discarded" for r in audit["records"]))

    def test_prespecified_selection_and_unknown_dual_ceiling(self):
        def run(hours, lower, upper=None, valid=True, method="sa_full"):
            return dict(hours=hours, reliable_lower=lower, coordination_upper=upper,
                        numerical_audit_passed=valid, method=method)
        selected = select_hours([run(1, 95.), run(2, 95.+1e-6), run(3, 97.), run(5, 99., valid=False)],
                                [run(3, 97., 98.)], 100.)
        self.assertEqual(selected["chosen_hours"], 1)
        self.assertEqual([r["reason"] for r in selected["excluded"]], ["certified_dual_ceiling", "numerical_audit"])
        selected = select_variant([run(1, 95., method="sa_flow"), run(1, 95.+1e-6, method="sa_compact"),
                                   run(1, 94., method="sa_full")])
        self.assertEqual(selected["chosen_variant"], "sa_flow")


if __name__ == "__main__":
    unittest.main()
