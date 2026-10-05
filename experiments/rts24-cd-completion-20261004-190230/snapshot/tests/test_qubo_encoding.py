import itertools
import unittest
from fractions import Fraction

import numpy as np

from suc_benders.master import _capacity_inequality, _quantize_inequality


def minimum_penalty(expression, fixed):
    """Enumerate the SDK expression's remaining bits, independently of our encoder."""
    free = sorted({name for term in expression.coefficient for name in term} - fixed.keys())
    if len(free) > 15:
        raise AssertionError("test constraint is too large to enumerate")
    indices = np.arange(1 << len(free))
    values = dict(fixed)
    values.update({name: (indices >> k) & 1 for k, name in enumerate(free)})
    energies = np.full(len(indices), float(expression.offset))
    for term, coefficient in expression.coefficient.items():
        product = coefficient
        for name in term:
            product = product * values[name]
        energies += product
    return float(energies.min())


class QuboEncodingTests(unittest.TestCase):
    def test_decimal_capacity_grid_preserves_every_binary_feasibility(self):
        capacity = {"a": Fraction("0.3"), "b": Fraction("0.2"), "c": Fraction("0.1")}
        for demand in (Fraction("0.5"), Fraction("0.500001"), Fraction("0")):
            encoded = _capacity_inequality(capacity, demand)
            for bits in itertools.product((0, 1), repeat=3):
                assignment = dict(zip(capacity, bits))
                actual = sum(value * assignment[key] for key, value in capacity.items()) >= demand
                residual = encoded.constant + sum(value * assignment[key] for key, value in encoded.coefficients.items())
                self.assertEqual(residual <= 0, actual)

    def test_quantized_cuts_only_relax_and_respect_error_budget(self):
        coefficients = {"a": -.123456789, "b": .234567891, "c": -1e-14}
        for budget in (1., .001):
            exact = {"theta": -Fraction(str(budget))}
            encoded = _quantize_inequality(coefficients, .1987654321, budget, exact)
            self.assertLessEqual(encoded.rounding_error, budget)
            for bits in itertools.product((0, 1), repeat=4):
                assignment = dict(zip(("a", "b", "c", "theta"), bits))
                original = Fraction(.1987654321)
                original += sum(Fraction(value) * assignment[key] for key, value in coefficients.items())
                original += exact["theta"] * assignment["theta"]
                rounded = encoded.step * (encoded.constant + sum(
                    value * assignment[key] for key, value in encoded.coefficients.items()
                ))
                self.assertLessEqual(rounded, original)
                self.assertLessEqual(float(original - rounded), budget)
                if original <= 0:
                    self.assertLessEqual(rounded, 0)


if __name__ == "__main__":
    unittest.main()
