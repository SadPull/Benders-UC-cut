import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import (
    decommit_commitment,
    make_candidate,
    on_blocks,
    repair_commitment,
)
from suc_benders.common import commitment_is_feasible
from suc_benders.data import UCData


ROOT = Path(__file__).resolve().parents[1]


class DecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def _start(self):
        """An over-committed but feasible starting commitment."""
        all_on = np.ones((self.units.n_units, self.units.horizon), dtype=int)
        base = make_candidate(self.units, self.scenarios, all_on, source="incumbent")
        self.assertTrue(
            commitment_is_feasible(
                self.units, self.scenarios, base.u, base.v, base.w
            )
        )
        return base

    def test_on_blocks_detects_spans(self):
        row = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
        self.assertEqual(on_blocks(row), [(1, 3), (6, 7), (9, 9)])
        self.assertEqual(on_blocks(np.zeros(5, dtype=int)), [])
        self.assertEqual(on_blocks(np.ones(4, dtype=int)), [(0, 3)])

    def test_ladder_is_feasible_and_monotone(self):
        base = self._start()
        ladder = decommit_commitment(
            self.units, self.scenarios, base, limit=8
        )
        self.assertTrue(ladder, "expected at least one decommitment rung")
        previous = int(base.u.sum())
        for rung in ladder:
            self.assertTrue(
                commitment_is_feasible(
                    self.units, self.scenarios, rung.u, rung.v, rung.w
                )
            )
            # Every rung removes on-hours relative to the previous rung.
            self.assertLess(int(rung.u.sum()), previous)
            previous = int(rung.u.sum())

    def test_ladder_rungs_are_unique(self):
        base = self._start()
        ladder = decommit_commitment(self.units, self.scenarios, base, limit=8)
        keys = {
            tuple(np.r_[r.u.ravel(), r.v.ravel(), r.w.ravel()].astype(int).tolist())
            for r in ladder
        }
        self.assertEqual(len(keys), len(ladder))

    def test_ladder_only_removes_from_on_blocks(self):
        base = self._start()
        ladder = decommit_commitment(self.units, self.scenarios, base, limit=8)
        for rung in ladder:
            # A decommitment can never turn a unit ON that was OFF in the base.
            self.assertTrue(np.all(rung.u <= base.u))

    def test_gradient_steers_move_order(self):
        base = self._start()
        # Strongly negative gradient on unit 9 hours 20-23 should make the
        # generator prioritise removing those hours.
        gradient = {(9, t): -100000.0 for t in range(20, 24)}
        ladder = decommit_commitment(
            self.units, self.scenarios, base, limit=1, gradient=gradient
        )
        self.assertTrue(ladder)
        move = ladder[0].move
        self.assertEqual(int(move["unit"]), 9)
        self.assertLessEqual(int(move["period"]), 23)
        self.assertGreaterEqual(int(move["period"]), 20 - 2)

    def test_gradient_counts_no_load_savings_once(self):
        # Each unit runs for two hours. Removing unit 0 saves 200 but adds
        # 160 in fuel; removing unit 1 saves 100 and adds no fuel.
        for startup, expected_unit in [([0.0, 0.0], 1), ([100.0, 0.0], 0)]:
            with self.subTest(startup=startup):
                data = UCData(
                    n_units=2, horizon=2,
                    pmax=np.full(2, 100.0), pmin=np.zeros(2),
                    pstart=np.full(2, 100.0), pshut=np.full(2, 100.0),
                    min_up=np.ones(2, dtype=int), min_down=np.ones(2, dtype=int),
                    ramp_up=np.full(2, 100.0), ramp_down=np.full(2, 100.0),
                    initial_on=np.zeros(2, dtype=int),
                    initial_duration=np.ones(2, dtype=int),
                    initial_output=np.zeros(2), base_load=np.zeros(2),
                    cost_a=np.zeros(2), cost_b=np.zeros(2),
                    cost_c=np.array([100.0, 50.0]),
                    startup_cost=np.array(startup), shutdown_cost=np.zeros(2),
                )
                base = make_candidate(data, [], np.ones((2, 2), dtype=int))
                gradient = {(i, t): change for i, change in enumerate([-20.0, -50.0])
                            for t in range(2)}
                ladder = decommit_commitment(data, [], base, limit=1, gradient=gradient)
                self.assertEqual(len(ladder), 1)
                expected = base.u.copy()
                expected[expected_unit, :] = 0
                np.testing.assert_array_equal(ladder[0].u, expected)

    def test_ladder_has_no_gurobi_dependency(self):
        import inspect

        import suc_benders.candidates as module

        source = inspect.getsource(module.decommit_commitment)
        self.assertNotIn("solve_scenario", source)
        self.assertNotIn("gurobipy", source)
        self.assertNotIn("require_gurobi", source)


if __name__ == "__main__":
    unittest.main()
