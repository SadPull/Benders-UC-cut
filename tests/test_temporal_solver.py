import itertools
import math
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from suc_benders.block_master_model import certify_master, partition_rows, priced_blocks
from suc_benders.coordination import converged, cutting_plane, coordinate
from suc_benders.data import Scenario
from suc_benders.paper_model import IndexedUC, CompactQubo, key
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.validation import validate_schedule
from tests.test_block_master_experiment import merged_optimum
from tests.test_capacity import one_hour_data


def tiny(horizon=3):
    data = replace(one_hour_data(), horizon=horizon, base_load=np.full(horizon, 30.))
    return data, [Scenario("tiny", 1., data.base_load)]


def enumerate_points(block):
    return [dict(zip(block.bkeys, bits)) for bits in itertools.product((0, 1), repeat=len(block.bkeys))
            if block.discrete_checks(dict(zip(block.bkeys, bits)))["feasible"]]


class EnumeratedQubo:
    """Replace only the sampling channel; real dispatch and certification run."""
    current = []

    def __init__(self, block, costs, cuts, lower, upper, config):
        type(self).current = enumerate_points(block)
        self.names, self.vars, self.weights = list(block.bkeys), {}, []
        self.lower, self.upper, self.energy_unit = lower, upper, 1.
        self.matrix = np.zeros((1, 1))

    def decode(self, point):
        return point, 0.


def enumerated_run(data, scenarios, config):
    with patch("suc_benders.paper_solver.CompactQubo", EnumeratedQubo), patch(
            "suc_benders.paper_solver.solve_qubo", side_effect=lambda *a, **k: EnumeratedQubo.current):
        return solve_benders(data, scenarios, config)


class TemporalSolverTests(unittest.TestCase):
    def test_partitions_one_two_three_hours_and_short_tail_recombine(self):
        data, scenarios = tiny(5)
        data = replace(data, min_up=np.array([4]), min_down=np.array([3]), initial_duration=np.array([4.]))
        uc = IndexedUC(data, scenarios)
        expected = merged_optimum(uc, priced_blocks(uc, 5, [0.]*len(uc.rows))[0], [])
        for hours in (1, 2, 3):
            with self.subTest(hours=hours):
                blocks, cross = priced_blocks(uc, hours, [0.]*len(uc.rows))
                self.assertEqual(sorted([i for b in blocks for i in b.row_ids]+cross), list(range(len(uc.rows))))
                self.assertEqual([t for b in blocks for t in b.times], list(range(5)))
                self.assertAlmostEqual(merged_optimum(uc, blocks, cross), expected, places=5)
                self.assertIn("min_up", {uc.rows[i].label for i in cross})

    def test_priced_bounds_planes_and_constants_by_enumeration(self):
        data, scenarios = tiny()
        uc = IndexedUC(data, scenarios)
        _, cross = partition_rows(uc, 1)
        prices = [0.]*len(uc.rows)
        for i in cross:
            prices[i] = -20. if uc.rows[i].equality else 2.
        blocks, _ = priced_blocks(uc, 1, prices)
        planes, bounds = [], []
        for block in blocks:
            results = [block.dispatch(x, block.costs, time.perf_counter()+30) for x in enumerate_points(block)]
            pool = [r for r in results if r and r["feasible"]]
            self.assertTrue(pool)
            optimum = min(r["cost"] for r in pool)+block.constant
            cert = certify_master(block, [r["cut"] for r in results if r], time.perf_counter()+30)
            self.assertEqual(set(cert), {"status", "lower_bound", "seconds"})
            self.assertLessEqual(cert["lower_bound"], optimum+1e-5)
            self.assertAlmostEqual(cert["lower_bound"], optimum, places=5)
            # Even a nonoptimal feasible candidate provides an upper plane.
            plane = cutting_plane(block, cross, max(pool, key=lambda r:r["cost"])["point"])
            planes.append([plane])
            bounds.append(cert["lower_bound"])
            for alternative in (prices, [0.]*len(prices)):
                other = priced_blocks(uc, 1, alternative)[0][len(planes)-1]
                other_pool = [other.dispatch(x, other.costs, time.perf_counter()+30) for x in enumerate_points(other)]
                true_dual = min(r["cost"] for r in other_pool if r["feasible"])+other.constant
                plane_value = plane["intercept"]+sum(v*alternative[i] for i,v in plane["slope"].items())
                self.assertGreaterEqual(plane_value+1e-5, true_dual)
        full = merged_optimum(uc, blocks, cross)
        coordinated = coordinate(uc, cross, planes, full, time.perf_counter()+30)
        self.assertGreaterEqual(coordinated["upper"]+1e-5, sum(bounds))
        self.assertLessEqual(coordinated["upper"], full+1e-5)

    def test_equality_multiplier_is_free(self):
        from types import SimpleNamespace
        uc = SimpleNamespace(rows=[SimpleNamespace(equality=True)])
        planes = [[dict(intercept=-2., slope={0:-1.}), dict(intercept=2., slope={0:1.})]]
        result = coordinate(uc, [0], planes, 10., time.perf_counter()+30)
        self.assertAlmostEqual(result["multipliers"][0], -2.)
        self.assertAlmostEqual(result["upper"], 0.)

    def test_tolerance_boundary_and_inconsistent_bounds(self):
        for upper in (100., -100.):
            tolerance = 1e-5+1e-4*abs(upper)
            self.assertTrue(converged(upper-tolerance+1e-10, upper, 1e-4))
            self.assertFalse(converged(upper-tolerance-1e-10, upper, 1e-4))
        self.assertFalse(converged(2., 1., 1e-4))
        self.assertFalse(converged(-math.inf, 1., 1e-4))

    def test_complete_nested_flow_with_enumerated_sampling(self):
        for hours in (1, 2, 3, 5):
            data, scenarios = tiny()
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=hours))
            self.assertEqual(result.status, "DUAL_CONVERGED", result.message)
            validate_schedule(data, scenarios, result)
            self.assertEqual(result.diagnostics["block_count"], math.ceil(3/hours))
            self.assertTrue(all(b["converged"] for b in result.diagnostics["blocks"]))
            self.assertLessEqual(result.diagnostics["reliable_lower"], result.objective+1e-5)

    def test_five_hour_blocks_cover_full_day_without_resetting_boundaries(self):
        data, scenarios = tiny(24)
        data = replace(data, min_up=np.array([4]), min_down=np.array([3]), initial_duration=np.array([1.]),
                       initial_on=np.array([True]), initial_output=np.array([30.]))
        uc = IndexedUC(data, scenarios)
        blocks, cross = priced_blocks(uc, 5, [0.]*len(uc.rows))
        self.assertEqual([len(b.times) for b in blocks], [5, 5, 5, 5, 4])
        self.assertEqual([t for b in blocks for t in b.times], list(range(24)))
        self.assertEqual(sorted([i for b in blocks for i in b.row_ids]+cross), list(range(len(uc.rows))))
        self.assertEqual(set(k for b in blocks for k in b.keys), set(uc.bounds))
        self.assertEqual({k[2] for b in blocks for r in b.rows if r.label == "initial" for k in r.a}, {0, 1, 2})
        self.assertEqual({k[2] for b in blocks for r in b.rows if r.label == "terminal" for k in r.a}, {23})
        self.assertTrue({"transition", "min_up", "min_down", "ramp_up", "ramp_down"} <= {uc.rows[i].label for i in cross})

    def test_five_hour_partition_with_four_hour_tail_matches_full_optimum(self):
        from suc_benders.full_model import FullConfig, solve_full
        data, scenarios = tiny(9)
        data = replace(data, min_up=np.array([4]), min_down=np.array([3]), initial_duration=np.array([1.]),
                       initial_on=np.array([True]), initial_output=np.array([30.]))
        uc = IndexedUC(data, scenarios)
        blocks, cross = priced_blocks(uc, 5, [0.]*len(uc.rows))
        self.assertEqual([len(b.times) for b in blocks], [5, 4])
        full = solve_full(data, scenarios, FullConfig(time_limit=30, mip_gap=1e-9, output_flag=0))
        try:
            self.assertAlmostEqual(merged_optimum(uc, blocks, cross), full.objective, places=5)
        finally:
            full.model.dispose()

    def test_raw_uvw_not_repaired_inside_block(self):
        data, scenarios = tiny()
        uc = IndexedUC(data, scenarios)
        raw = {k:0 for k in priced_blocks(uc, 3, [0.]*len(uc.rows))[0][0].bkeys}
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[raw]), patch(
                "suc_benders.paper_solver.CompactQubo", EnumeratedQubo):
            result = solve_benders(data, scenarios, BendersConfig(max_inner_iterations=1))
        self.assertFalse(result.diagnostics["proposals"][0]["accepted"])
        self.assertFalse(any(r["origin"] == "sa" for r in result.diagnostics["local_evaluations"]))

    def test_duplicate_does_not_converge_and_no_coordination_on_inner_limit(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.certify_master", return_value=dict(lower_bound=0., seconds=0., status=2)), patch(
                "suc_benders.paper_solver.converged", return_value=False), patch(
                "suc_benders.paper_solver.coordinate") as outer:
            result = solve_benders(data, scenarios, BendersConfig(max_inner_iterations=3))
        self.assertEqual(result.status, "INNER_NOT_CONVERGED")
        self.assertEqual(result.diagnostics["sa_calls"], 3)
        self.assertTrue(math.isfinite(result.objective))
        outer.assert_not_called()

    def test_missing_certification_bound_reports_stage(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.solve_qubo", return_value=[]), patch(
                "suc_benders.paper_solver.certify_master", return_value=dict(lower_bound=None, seconds=0., status=9)):
            result = solve_benders(data, scenarios, BendersConfig())
        self.assertEqual(result.status, "CERTIFICATION_FAILED")
        self.assertIn("certification", result.diagnostics["failure_stage"])
        self.assertTrue(math.isfinite(result.objective))

    def test_integer_primal_outputs_cannot_enter_search(self):
        data, scenarios = tiny()
        original = certify_master
        def poisoned(*args):
            return dict(original(*args), point={"invalid": object()}, objective=-1e99, theta=-1e99)
        baseline = enumerated_run(data, scenarios, BendersConfig(block_hours=1))
        with patch("suc_benders.paper_solver.certify_master", side_effect=poisoned):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1))
        np.testing.assert_array_equal(result.u, baseline.u)
        self.assertEqual(result.objective, baseline.objective)
        self.assertEqual(result.optimality_cuts, baseline.optimality_cuts)
        self.assertEqual(len(result.diagnostics["local_evaluations"]), len(baseline.diagnostics["local_evaluations"]))
        self.assertTrue(all(r["origin"] in {"incumbent_projection", "previous_candidate", "sa"} for r in result.diagnostics["local_evaluations"]))

    def test_price_change_rebuilds_cuts_and_redispatches_old_candidates(self):
        data, scenarios = tiny()
        seen = []
        def certificate(block, cuts, deadline):
            seen.append(cuts)
            return certify_master(block, cuts, deadline)
        real_converged = converged
        def inner_only(*args, **kwargs):
            return False if "reference" in kwargs else real_converged(*args, **kwargs)
        with patch("suc_benders.paper_solver.certify_master", side_effect=certificate), patch(
                "suc_benders.paper_solver.converged", side_effect=inner_only):
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1, max_iterations=2))
        self.assertEqual(result.status, "MAX_ITERATIONS")
        self.assertEqual(len(result.diagnostics["outer"]), 2)
        self.assertTrue(any(r["outer"] == 2 for r in result.diagnostics["local_evaluations"]))
        for first, second in zip(seen[:3], seen[3:]):
            self.assertIsNot(first, second)
            self.assertFalse({id(c) for c in first} & {id(c) for c in second})

    def test_timeout_cancellation_and_evaluation_limit_are_not_convergence(self):
        data, scenarios = tiny()
        result = solve_benders(data, scenarios, BendersConfig(time_limit_seconds=1e-12))
        self.assertEqual(result.status, "TIME_LIMIT")
        with patch("suc_benders.paper_solver.cancelled", return_value=True):
            result = solve_benders(data, scenarios, BendersConfig())
        self.assertEqual(result.status, "CANCELLED")
        result = solve_benders(data, scenarios, BendersConfig(max_candidate_evaluations=1))
        self.assertEqual(result.status, "EVALUATION_LIMIT")
        self.assertTrue(math.isfinite(result.objective))

    def test_retired_options_and_infinite_outer_are_rejected(self):
        data, scenarios = tiny()
        for config in (BendersConfig(target_cost=1.), BendersConfig(stop_on_no_new_candidates=False),
                       BendersConfig(max_iterations=None), BendersConfig(block_hours=6)):
            with self.assertRaises(ValueError):
                solve_benders(data, scenarios, config)

    def test_initialization_and_coordination_failures_preserve_stage(self):
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.continuous_relaxation", side_effect=RuntimeError("pricing failed")):
            result = solve_benders(data, scenarios, BendersConfig())
        self.assertEqual(result.status, "INITIALIZATION_FAILED")
        self.assertEqual(result.diagnostics["failure_stage"], "initialization")
        with patch("suc_benders.paper_solver.coordinate", side_effect=RuntimeError("LP failed")):
            result = enumerated_run(data, scenarios, BendersConfig())
        self.assertEqual(result.status, "COORDINATION_FAILED")
        self.assertTrue(math.isfinite(result.objective))
        self.assertTrue(math.isfinite(result.diagnostics["reliable_lower"]))

    def test_constant_is_removed_from_qubo_and_restored_in_certificate(self):
        data, scenarios = tiny(1)
        uc = IndexedUC(data, scenarios)
        block = priced_blocks(uc, 1, [0.]*len(uc.rows))[0][0]
        result = block.dispatch(enumerate_points(block)[0], block.costs, time.perf_counter()+30)
        cuts = [result["cut"]]
        lower = block.relaxation(block.costs, cuts, time.perf_counter()+30)-1e-5
        original = CompactQubo(block, block.costs, cuts, lower, result["cost"], BendersConfig())
        before = certify_master(block, cuts, time.perf_counter()+30)
        block.constant = -123.25
        after = certify_master(block, cuts, time.perf_counter()+30)
        shifted = CompactQubo(block, block.costs, cuts,
                              (lower+block.constant)-block.constant,
                              (result["cost"]+block.constant)-block.constant, BendersConfig())
        self.assertAlmostEqual(after["lower_bound"]-before["lower_bound"], block.constant)
        np.testing.assert_allclose(original.matrix, shifted.matrix)

    def test_llm_only_quota_is_resource_stop_even_when_candidates_are_optimal(self):
        from suc_benders.llm import LLMConfig, LLMAdviceResult
        data, scenarios = tiny()
        with patch("suc_benders.paper_solver.ask_llm_for_candidates", return_value=LLMAdviceResult([])):
            result = solve_benders(data, scenarios, BendersConfig(advisor_only=True,
                quantum_candidates=False, llm=LLMConfig(enabled=True, max_calls=1)))
        self.assertEqual(result.status, "LLM_CALL_LIMIT")
        self.assertTrue(math.isfinite(result.objective))
        self.assertEqual(result.diagnostics["sa_calls"], 0)

    def test_advisor_call_numbers_increase_across_blocks_and_outer_iterations(self):
        from suc_benders.llm import LLMConfig, LLMAdviceResult
        data, scenarios = tiny()
        def inner_only(*args, **kwargs):
            return False if "reference" in kwargs else converged(*args, **kwargs)
        with patch("suc_benders.paper_solver.converged", side_effect=inner_only), patch(
                "suc_benders.paper_solver.ask_llm_for_candidates", return_value=LLMAdviceResult([])) as ask:
            result = enumerated_run(data, scenarios, BendersConfig(block_hours=1,
                max_iterations=2, llm=LLMConfig(enabled=True)))
        self.assertEqual(result.status, "MAX_ITERATIONS")
        self.assertEqual([call.args[2] for call in ask.call_args_list], list(range(1, 7)))


if __name__ == "__main__":
    unittest.main()
