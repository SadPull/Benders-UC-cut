import unittest
from unittest.mock import patch

import numpy as np

from suc_benders.master import QuantumConfig
from suc_benders.quantum import solve_qubo


class QuantumSolverTests(unittest.TestCase):
    def test_auxiliary_spin_sign_does_not_change_binary_solution(self):
        spins = np.array([[1, -1, 1], [-1, 1, -1]])
        with patch("kaiwu.classical.SimulatedAnnealingOptimizer.solve", return_value=spins):
            solutions = solve_qubo(np.diag([-1.0, 2.0]), QuantumConfig())
        for solution in solutions:
            np.testing.assert_array_equal(solution, [1, 0])

    def test_local_annealing_finds_small_qubo_minimum(self):
        matrix = np.diag([-1.0, 2.0])
        solutions = solve_qubo(matrix, QuantumConfig(
            initial_temperature=10, alpha=0.8, cutoff_temperature=0.01,
            iterations_per_temperature=10, candidate_pool_size=4,
        ))
        self.assertTrue(solutions)
        self.assertEqual(min(float(row @ matrix @ row) for row in solutions), -1.0)


if __name__ == "__main__":
    unittest.main()
