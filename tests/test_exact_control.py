import itertools
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from suc_benders.data import Scenario
from suc_benders.exact_control import Budget, aggregate, closed_loop, solve_block, unpack
from suc_benders.exact_experiment import checkpoint, offline, read, verify_source
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.paper_model import Block, IndexedUC
from tests.test_capacity import one_hour_data


class ExactControlTests(unittest.TestCase):
    def setUp(self):
        self.data = one_hour_data()
        self.scenarios = [Scenario("a", .25, self.data.base_load),
                          Scenario("b", .75, self.data.base_load*.9)]

    def test_enumeration_and_full_model(self):
        uc = IndexedUC(self.data, self.scenarios, 3)
        block = Block(uc, 0)
        result = solve_block(block, uc.linear, time.perf_counter()+30)
        values = []
        for bits in itertools.product((0, 1), repeat=len(block.bkeys)):
            x = dict(zip(block.bkeys, bits))
            if block.discrete_feasible(x):
                r = block.dispatch(x, uc.linear, time.perf_counter()+30)
                if r and r["feasible"]:
                    values.append(r["cost"])
        full = solve_full(self.data, self.scenarios, FullConfig(time_limit=30, mip_gap=1e-8, output_flag=0))
        self.assertTrue(result["certified"])
        self.assertAlmostEqual(result["objective"], min(values), places=4)
        self.assertAlmostEqual(result["objective"], full.objective, places=4)
        self.assertLessEqual(result["bound"], min(values)+1e-4)

    def test_nonzero_prices_and_global_bound(self):
        uc = IndexedUC(self.data, self.scenarios, 1)
        prices = np.full(len(uc.coupling), .001)
        costs, constant = uc.coefficients(prices)
        records = [solve_block(Block(uc, j), costs, time.perf_counter()+30) for j in range(3)]
        result = aggregate(records, 3, constant)
        point = {k: v for r in records for k, v in unpack(r["point"]).items()}
        self.assertAlmostEqual(result["upper"], uc.original_cost(point)+uc.cost_scale*np.dot(prices, uc.residuals(point)), places=5)
        full = solve_full(self.data, self.scenarios, FullConfig(output_flag=0))
        self.assertLessEqual(result["lower"], full.objective+1e-4)

    def test_expired_block_does_not_open_solver(self):
        uc = IndexedUC(self.data, self.scenarios, 1)
        with patch("suc_benders.exact_control.require_gurobi", side_effect=AssertionError("solver called")):
            r = solve_block(Block(uc, 0), uc.linear, time.perf_counter()-1)
        self.assertEqual(r["status"], "BUDGET_EXHAUSTED")
        self.assertIsNone(r["point"])

    def test_partial_and_uncertified_bounds(self):
        record = dict(bound=5., objective=7., certified=False)
        self.assertEqual(aggregate([record], 1, 3.), dict(lower=8., upper=10., certified=False))
        self.assertIsNone(aggregate([record], 2, 3.)["lower"])
        self.assertIsNone(aggregate([{**record, "objective": None}], 1, 3.)["upper"])
        self.assertEqual(aggregate([{**record, "objective": None}], 1, 3.)["lower"], 8.)
        self.assertIsNone(aggregate([{**record, "bound": None}], 1, 3.)["lower"])

    def test_time_limit_preserves_bound_without_claiming_optimum(self):
        class Variable(float):
            @property
            def X(self):
                return float(self)

        class Model:
            Params = SimpleNamespace()
            Status = 9
            ObjBound = -1.
            ObjVal = 0.
            MIPGap = 1.
            def addVar(self, **kwargs):
                return Variable(0.)
            def setObjective(self, *args):
                pass
            def optimize(self):
                pass
            def dispose(self):
                pass

        codes = SimpleNamespace(BINARY=1, CONTINUOUS=2, MINIMIZE=1, OPTIMAL=2,
                                TIME_LIMIT=9, INFEASIBLE=3, INF_OR_UNBD=4, INTERRUPTED=11)
        k = ("u", 0, 0, -1)
        block = SimpleNamespace(keys=[k], bkeys=[k], rows=[],
                                uc=SimpleNamespace(bounds={k: (0., 1.)}, quadratic={}))
        for count in (0, 1):
            model = Model()
            model.SolCount = count
            gp = SimpleNamespace(Model=lambda _: model, GRB=codes, quicksum=sum)
            with patch("suc_benders.exact_control.require_gurobi", return_value=gp):
                result = solve_block(block, {k: 1.}, time.perf_counter()+30)
            self.assertEqual(result["status"], "TIME_LIMIT")
            self.assertEqual(result["bound"], -1.)
            self.assertFalse(result["certified"])
            self.assertEqual(result["objective"], 0. if count else None)
            self.assertEqual(result["point"] is not None, bool(count))

    def test_nonfinite_solver_bound_is_not_aggregated(self):
        from suc_benders.exact_control import finite
        for value in (float("nan"), float("inf"), -float("inf"), 1e100):
            self.assertIsNone(finite(value))

    def test_budget_restart_charges_active_downtime(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            state = {"budget": {"used": 12., "active_since": 100.}}
            with patch("suc_benders.exact_control.time.time", return_value=108.), \
                 patch("suc_benders.exact_control.time.perf_counter", return_value=50.):
                b = Budget(p, state, 30.)
                self.assertEqual(b.deadline, 60.)
                b.save(finished=True)
            saved = read(p)
            self.assertEqual(saved["budget"]["used"], 20.)
            with patch("suc_benders.exact_control.time.perf_counter", return_value=70.):
                b = Budget(p, saved, 30.)
                self.assertEqual(b.deadline, 80.)

    def test_exhausted_resume_does_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {"budget": {"used": 30., "active_since": None}}
            b = Budget(Path(directory)/"s.json", state, 30.)
            self.assertFalse(b.available())

    def test_closed_loop_isolated_and_physically_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            budget = Budget(Path(directory)/"closed.json", state, 30.)
            with patch("suc_benders.quantum.solve_qubo", side_effect=AssertionError("SA called")), \
                 patch("suc_benders.paper_solver.solve_qubo", side_effect=AssertionError("SA called")), \
                 patch("suc_benders.paper_model.get_kaiwu", side_effect=AssertionError("QUBO called")), \
                 patch("suc_benders.solver.ask_llm_for_candidates", side_effect=AssertionError("LLM called")):
                closed_loop(self.data, self.scenarios, 1, state, budget, max_rounds=2)
            self.assertTrue(state["finished"])
            self.assertTrue(state["evaluations"])
            self.assertTrue(all(e["validation"]["feasible_solution"] for e in state["evaluations"]))
            self.assertLessEqual(len(state["rounds"]), 2)

    def test_closed_incomplete_round_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            b = Budget(Path(directory)/"closed.json", state, 30.)
            empty = dict(bound=None, objective=None, point=None, certified=False)
            with patch("suc_benders.exact_control.solve_block", return_value=empty):
                closed_loop(self.data, self.scenarios, 1, state, b)
            self.assertEqual(state["status"], "INCOMPLETE_ROUND")
            self.assertIsNone(state["best_lower"])

    def test_offline_cache_and_no_source_mutation(self):
        uc = IndexedUC(self.data, self.scenarios, 1)
        job = dict(id="one", hours=1, prices=[0.]*len(uc.coupling), cache_key="zero", sampled=10000.)
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            b = Budget(Path(directory)/"offline.json", state, 30.)
            with patch("suc_benders.exact_experiment.solve_block", wraps=solve_block) as solve:
                offline(self.data, self.scenarios, [job, {**job, "id": "two"}], state, b, 30.)
                self.assertEqual(solve.call_count, 3)
            self.assertEqual(len(state["snapshots"]), 2)
            self.assertTrue(state["snapshots"][0]["certified"])

    def test_offline_exhausted_marks_all(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            b = Budget(Path(directory)/"offline.json", state, 0.)
            job = dict(id="one", cache_key="zero")
            with patch("suc_benders.exact_experiment.solve_block", side_effect=AssertionError("solver called")):
                offline(self.data, self.scenarios, [job], state, b, 30.)
            self.assertEqual(state["snapshots"][0]["status"], "BUDGET_EXHAUSTED")

    def test_offline_resume_reuses_pending_blocks(self):
        uc = IndexedUC(self.data, self.scenarios, 1)
        costs, _ = uc.coefficients(np.zeros(len(uc.coupling)))
        record = solve_block(Block(uc, 0), costs, time.perf_counter()+30)
        job = dict(id="one", hours=1, prices=[0.]*len(uc.coupling), cache_key="zero", sampled=10000.)
        with tempfile.TemporaryDirectory() as directory:
            state = {"pending": {"key": "zero", "records": [record]}}
            b = Budget(Path(directory)/"offline.json", state, 30.)
            with patch("suc_benders.exact_experiment.solve_block", wraps=solve_block) as solve:
                offline(self.data, self.scenarios, [job], state, b, 30.)
                self.assertEqual(solve.call_count, 2)
            self.assertTrue(state["snapshots"][0]["certified"])

    def test_checkpoints_exclude_late_results(self):
        state = dict(evaluations=[dict(elapsed=10., objective=100.), dict(elapsed=301., objective=80.)],
                     rounds=[dict(bound_elapsed=299., best_lower=70.), dict(bound_elapsed=301., best_lower=75.)])
        self.assertEqual(checkpoint(state, 300.), dict(seconds=300., objective=100., lower=70.))

    def test_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/"manifest.json").write_text('{"sources":{"suc_benders/paper_model.py":"wrong"}}')
            with self.assertRaisesRegex(ValueError, "source model mismatch"):
                verify_source(p, self.data, self.scenarios)


if __name__ == "__main__":
    unittest.main()
