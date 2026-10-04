import unittest
from pathlib import Path

import numpy as np

from suc_benders import load_suc_data
from suc_benders.candidates import make_candidate, repair_commitment
from suc_benders.common import commitment_is_feasible


ROOT = Path(__file__).resolve().parents[1]


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/rts24/units.json")

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
        self.assertIsNone(repair_commitment(self.units, self.scenarios, np.full((self.units.n_units, self.units.horizon), np.nan)))

    def test_make_candidate_rebuilds_transitions(self):
        u = np.zeros((self.units.n_units, self.units.horizon), dtype=int)
        candidate = make_candidate(self.units, self.scenarios, u)
        previous = np.column_stack((self.units.initial_on.astype(int), candidate.u[:, :-1]))
        np.testing.assert_array_equal(candidate.u - previous, candidate.v - candidate.w)
        np.testing.assert_array_equal(candidate.v + candidate.w <= 1, np.ones_like(candidate.v, dtype=bool))


if __name__ == "__main__":
    unittest.main()
