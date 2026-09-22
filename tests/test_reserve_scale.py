import unittest
from pathlib import Path

from suc_benders import load_suc_data
from suc_benders.master import QuantumMaster


ROOT = Path(__file__).resolve().parents[1]


class ReserveScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.scenarios = load_suc_data(ROOT / "data/suc_10x24.json")

    def test_reserve_scale_changes_matrix_but_not_variable_count(self):
        base = QuantumMaster(self.units, self.scenarios, [], theta_cap=1024, reserve_scale=1.0)
        scaled = QuantumMaster(self.units, self.scenarios, [], theta_cap=1024, reserve_scale=100.0)
        self.assertEqual(base.variable_count(), scaled.variable_count())
        self.assertFalse(__import__("numpy").array_equal(base.matrix(), scaled.matrix()))


if __name__ == "__main__":
    unittest.main()
