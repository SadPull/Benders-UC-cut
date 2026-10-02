import itertools
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from suc_benders.candidates import make_candidate
from suc_benders.common import commitment_is_feasible
from suc_benders.data import Scenario
from suc_benders.master import QuantumConfig
from suc_benders.paper_model import IndexedUC, Block, CompactQubo, key, binary, name, grid_bounds
from suc_benders.solver import BendersConfig, solve_benders
from suc_benders.subproblem import solve_scenario
from tests.test_capacity import one_hour_data
from tests.test_qubo_encoding import minimum_penalty


def candidate_point(c):
    return {key(kind, i, t): int(getattr(c, kind)[i, t]) for kind in ("u", "v", "w")
            for i in range(c.u.shape[0]) for t in range(c.u.shape[1])}


class PaperModelTests(unittest.TestCase):
    def setUp(self):
        self.data = one_hour_data()
        self.scenarios = [Scenario("tiny", 1., self.data.base_load)]

    def test_full_horizon_preserves_initial_and_cross_hour_constraints(self):
        data = replace(self.data, horizon=5, base_load=np.array([30., 30., 30., 0., 0.]),
            min_up=np.array([4]), min_down=np.array([3]),
            initial_on=np.array([True]), initial_duration=np.array([1.]))
        uc = IndexedUC(data, [Scenario("tiny", 1., data.base_load)])
        block = Block(uc)
        self.assertEqual(block.times, (0, 1, 2, 3, 4))
        self.assertEqual([k[2] for row in block.rows if row.label == "terminal" for k in row.a], [4])
        self.assertEqual([k[2] for row in block.rows if row.label == "initial" for k in row.a], [0, 1, 2])
        self.assertTrue(any(row.label == "min_up" and
            max(k[2] for k in row.a)-min(k[2] for k in row.a) >= 3 for row in block.rows))

    def test_one_hour_capacity_penalty_preserves_separate_limits(self):
        for demand, feasible in ((35., True), (35.000001, False)):
            data = replace(self.data, base_load=np.array([demand, 0., 0.]))
            scenarios = [Scenario("tiny", 1., data.base_load)]
            uc = IndexedUC(data, scenarios)
            candidate = make_candidate(data, scenarios, np.array([[1, 0, 0]]))
            for scale in (1., 100.):
                master = CompactQubo(Block(uc), uc.linear, [], 0., 500., BendersConfig(reserve_scale=scale))
                values = {name(k): value for k, value in candidate_point(candidate).items()}
                values.update({"and_0_0": 1, "and_0_1": 0})
                penalty = minimum_penalty(master.model.hard_constraints_made["reserve_0"].constraint_expr, values)
                self.assertEqual(penalty < 1e-12, feasible)


    def test_all_binary_rows_equal_original_discrete_rules(self):
        d = replace(self.data, base_load=np.zeros(3))
        ss = [Scenario("zero", 1., d.base_load)]
        uc = IndexedUC(d, ss)
        rows = [r for r in uc.rows if all(binary(k) for k in r.a)]
        ks = [k for k in uc.bounds if binary(k)]
        for bits in itertools.product((0, 1), repeat=len(ks)):
            x = dict(zip(ks, bits))
            arrays = [np.array([[x[key(kind, 0, t)] for t in range(3)]]) for kind in ("u", "v", "w")]
            valid = all(abs(r.value(x)) < 1e-7 if r.equality else r.value(x) < 1e-7 for r in rows)
            self.assertEqual(valid, commitment_is_feasible(d, ss, *arrays))

    def test_joint_qp_matches_original_and_cuts_support_every_feasible_point(self):
        uc = IndexedUC(self.data, self.scenarios)
        block = Block(uc)
        records = []
        for bits in itertools.product((0, 1), repeat=3):
            candidate = make_candidate(self.data, self.scenarios, np.array([bits]))
            x = candidate_point(candidate)
            if not block.discrete_feasible(x):
                continue
            result = block.dispatch(x, uc.linear, time.perf_counter()+30)
            original = solve_scenario(self.data, self.scenarios[0], candidate.u, candidate.v, candidate.w)
            self.assertEqual(result["feasible"], original.feasible)
            if original.feasible:
                self.assertAlmostEqual(result["recourse"], original.objective, places=5)
                for row in uc.rows:
                    self.assertLessEqual(abs(row.value(result["point"])) if row.equality else row.value(result["point"]), 1e-5)
            records.append((x, result))
        feasible = [(x, r) for x, r in records if r["feasible"]]
        self.assertTrue(feasible)
        for x, result in records:
            cut = result["cut"]
            if result["feasible"]:
                self.assertAlmostEqual(cut.value(x), result["recourse"], places=5)
            else:
                self.assertGreater(cut.value(x), 1e-5)
            for xx, rr in feasible:
                self.assertLessEqual(cut.value(xx), (rr["recourse"] if cut.feasible else 0.)+1e-5)
        lb = block.relaxation(uc.linear, [r["cut"] for _, r in records], time.perf_counter()+30)
        self.assertLessEqual(lb, min(r["cost"] for _, r in feasible)+1e-5)


    def test_compact_integer_penalties_and_fixed_decoding(self):
        uc = IndexedUC(self.data, self.scenarios)
        block = Block(uc)
        config = BendersConfig()
        master = CompactQubo(block, uc.linear, [], 0., 500., config)
        ks = [k for k in block.bkeys if k not in block.fixed]
        for bits in itertools.product((0, 1), repeat=len(ks)):
            x = {**block.fixed, **dict(zip(ks, bits))}
            values = {name(k): v for k, v in x.items()}
            energy = sum(minimum_penalty(c.constraint_expr, values) for label, c in master.model.hard_constraints_made.items() if label.startswith("hard_"))
            self.assertEqual(abs(energy) < 1e-7, block.discrete_feasible(x))
        decoded, theta = master.decode(np.zeros(len(master.names), dtype=int))
        self.assertTrue(all(decoded[k] == v for k, v in block.fixed.items()))
        self.assertAlmostEqual(theta+sum(uc.linear[k]*decoded[k] for k in block.bkeys), master.lower)

    def test_negative_and_zero_width_encoding(self):
        self.assertEqual(grid_bounds(-3., -3., 1.), (-3., -3.))
        lo, hi = grid_bounds(-9.2, -2.4, 1.)
        self.assertLessEqual(lo, -9.2)
        self.assertGreaterEqual(hi, -2.4)
        uc = IndexedUC(self.data, self.scenarios)
        block = Block(uc)
        master = CompactQubo(block, uc.linear, [], -9.2, -2.4, BendersConfig())
        zero_width = CompactQubo(block, uc.linear, [], -3., -3., BendersConfig())
        self.assertEqual(zero_width.weights, [])
        possible = {master.lower+sum(w*b for w, b in zip(master.zvars.values(), bits))
                    for bits in itertools.product((0, 1), repeat=len(master.weights))}
        self.assertEqual(possible, set(np.arange(master.lower, master.upper+1)))
        with self.assertRaises(ValueError):
            grid_bounds(5, 3, 1)

    def test_scenario_probabilities_are_applied_once(self):
        scenarios = [Scenario("a", .2, self.data.base_load), Scenario("b", .8, self.data.base_load)]
        uc = IndexedUC(self.data, scenarios)
        c = make_candidate(self.data, scenarios, np.array([[1, 0, 0]]))
        r = Block(uc).dispatch(candidate_point(c), uc.linear, time.perf_counter()+30)
        expected = solve_scenario(self.data, scenarios[0], c.u, c.v, c.w)
        self.assertAlmostEqual(r["recourse"], expected.objective, places=5)

    def test_whole_horizon_search_uses_only_continuous_gurobi_models(self):
        self.data = replace(self.data, base_load=np.full(3, 30.))
        self.scenarios = [Scenario("tiny", 1., self.data.base_load)]
        original = gp.Model.optimize
        seen = []
        def guard(m, *args, **kwargs):
            m.update()
            self.assertTrue(all(v.VType == gp.GRB.CONTINUOUS for v in m.getVars()))
            seen.append(m.ModelName)
            return original(m, *args, **kwargs)
        config = BendersConfig(
            max_iterations=2,
            time_limit_seconds=20, max_candidate_evaluations=40,
            quantum=QuantumConfig(candidate_pool_size=2, alpha=.8, iterations_per_temperature=1))
        with patch.object(gp.Model, "optimize", guard), \
             patch("suc_benders.full_model.solve_full", side_effect=AssertionError("reference")), \
             patch("suc_benders.paper_solver.ask_llm_for_candidates", side_effect=AssertionError("LLM")):
            result = solve_benders(self.data, self.scenarios, config)
        self.assertIn("paper_master_relaxation", seen)
        self.assertTrue(np.isfinite(result.objective))
        self.assertTrue(commitment_is_feasible(self.data, self.scenarios, result.u, result.v, result.w))
        self.assertEqual(result.diagnostics["evaluation_count"], len(result.diagnostics["evaluations"]))

    def test_whole_horizon_reuses_local_evaluations_at_unchanged_prices(self):
        vectors = {k: np.repeat(v, 2) for k, v in vars(self.data).items()
                   if isinstance(v, np.ndarray) and k != "base_load"}
        d = replace(self.data, **vectors, n_units=2, base_load=np.full(3, 30.))
        ss = [Scenario("tiny", 1., d.base_load)]
        config = BendersConfig(
                               max_iterations=3, time_limit_seconds=20.)
        # First round adds a cheaper one-unit schedule, forcing a second round.
        # Subsequent samples repeat it; both old and new seeds must hit cache.
        def repeat_seed(master, bits):
            c = make_candidate(d, ss, np.array([[1, 1, 1], [0, 0, 0]]))
            return candidate_point(c), 0.
        with patch.object(CompactQubo, "decode", repeat_seed):
            result = solve_benders(d, ss, config)
        self.assertGreaterEqual(len(result.history), 2)
        self.assertEqual(result.diagnostics["local_evaluation_count"], 2)
        self.assertEqual(result.optimality_cuts, 2)


if __name__ == "__main__":
    unittest.main()
