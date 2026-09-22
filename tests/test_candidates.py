import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import make_candidate, perturb_commitment, repair_commitment
from suc_benders.common import commitment_is_feasible


ROOT = Path(__file__).resolve().parents[1]


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_repair_invalid_input_without_mutation(self):
        raw = np.zeros((self.units.n_units, self.units.horizon), dtype=int)
        repaired = repair_commitment(self.units, self.scenarios, raw)
        self.assertIsNotNone(repaired)
        self.assertTrue(
            commitment_is_feasible(self.units, self.scenarios, repaired.u, repaired.v, repaired.w)
        )
        self.assertTrue(np.all(raw == 0))

    def test_repair_rejects_bad_shape(self):
        self.assertIsNone(repair_commitment(self.units, self.scenarios, np.zeros((1, 1))))
        self.assertIsNone(repair_commitment(self.units, self.scenarios, np.full((10, 24), np.nan)))

    def test_make_candidate_rebuilds_transitions(self):
        u = np.zeros((self.units.n_units, self.units.horizon), dtype=int)
        candidate = make_candidate(self.units, self.scenarios, u)
        previous = np.column_stack((self.units.initial_on.astype(int), candidate.u[:, :-1]))
        np.testing.assert_array_equal(candidate.u - previous, candidate.v - candidate.w)
        np.testing.assert_array_equal(candidate.v + candidate.w <= 1, np.ones_like(candidate.v, dtype=bool))

    def test_perturbations_are_feasible_and_distinct(self):
        base = repair_commitment(self.units, self.scenarios, np.zeros((self.units.n_units, self.units.horizon), dtype=int))
        self.assertIsNotNone(base)
        neighbors = perturb_commitment(
            self.units,
            self.scenarios,
            base,
            limit=16,
            window_lengths=(2, 3, 4),
            group_size=2,
            seed=7,
        )
        self.assertGreaterEqual(len(neighbors), 8)
        keys = [
            tuple(np.r_[c.u.ravel(), c.v.ravel(), c.w.ravel()].astype(int).tolist())
            for c in neighbors
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(
            all(
                commitment_is_feasible(self.units, self.scenarios, c.u, c.v, c.w)
                for c in neighbors
            )
        )


if __name__ == "__main__":
    unittest.main()
