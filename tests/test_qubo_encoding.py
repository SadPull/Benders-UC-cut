import itertools
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from suc_benders.candidates import make_candidate
from suc_benders.common import commitment_capacity, commitment_is_feasible
from suc_benders.data import Scenario, load_suc_data
from suc_benders.master import (
    BendersCut, QuantumMaster, _capacity_inequality, _quantize_inequality,
)
from suc_benders.solver import _fixed_cost
from suc_benders.subproblem import aggregate_optimality_cut, solve_scenario
from tests.test_capacity import one_hour_data


ROOT = Path(__file__).resolve().parents[1]


def fixed_bits(master, candidate):
    values = {f"{name}_{i}_{t}": int(getattr(candidate, name)[i, t])
              for name in ("u", "v", "w") for i in range(master.data.n_units)
              for t in range(master.data.horizon)}
    values.update({f"theta_{k}": 0 for k in range(master.theta_bits)})
    values.update({f"_one_hour_{i}_{t}": int(candidate.v[i, t] * candidate.w[i, t + 1])
                   for i in range(master.data.n_units) for t in range(master.data.horizon - 1)})
    return values


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

    def test_one_hour_auxiliary_matches_separate_limits(self):
        data = one_hour_data()
        scenario = Scenario("small", 1., data.base_load)
        master = QuantumMaster(data, [scenario], [], theta_cap=100.)
        master.matrix()
        self.assertNotIn("_one_hour_0_0", master.decision_variables())
        self.assertIn("_one_hour_0_0", master.qubo_variables())
        for v, w, z in itertools.product((0, 1), repeat=3):
            constraint = master.model.hard_constraints_made["_one_hour_0_0"].constraint_expr
            value = constraint.get_val({"v_0_0": v, "w_0_1": w, "_one_hour_0_0": z})
            self.assertEqual(value == 0, z == v * w)
        candidate = make_candidate(data, [scenario], np.array([[1, 0, 0]]))
        values = fixed_bits(master, candidate)
        for name, constraint in master.model.hard_constraints_made.items():
            if name.startswith("reserve_"):
                self.assertAlmostEqual(minimum_penalty(constraint.constraint_expr, values), 0., places=9)
        encoded = master._encoded_inequalities["reserve_0_0"]
        actual_capacity = -encoded.step * sum(
            value * values[key] for key, value in encoded.coefficients.items()
        )
        self.assertEqual(float(actual_capacity), commitment_capacity(data, candidate.u, candidate.v, candidate.w)[0, 0])

    def test_default_feasible_schedules_have_zero_penalty_and_correct_order(self):
        data, scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        all_on = make_candidate(data, scenarios, np.ones((data.n_units, data.horizon), dtype=int))
        u = all_on.u.copy()
        u[4, :5] = 0
        shoulder = make_candidate(data, scenarios, u)
        energies = []
        for scale in (1., 100.):
            master = QuantumMaster(data, scenarios, [], theta_cap=1e6, reserve_scale=scale)
            master.matrix()
            pair = []
            for candidate in (all_on, shoulder):
                self.assertTrue(commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w))
                values = fixed_bits(master, candidate)
                penalty = sum(
                    minimum_penalty(constraint.constraint_expr, values) * constraint.penalty
                    for constraint in master.model.hard_constraints_made.values()
                )
                self.assertAlmostEqual(penalty, 0., delta=1e-5)
                objective = master.model.objective.get_val(values)
                self.assertAlmostEqual(objective * master.cost_scale, _fixed_cost(data, candidate.u, candidate.v, candidate.w))
                pair.append(objective + penalty)
            self.assertLess(pair[1], pair[0])
            energies.append(pair)
        np.testing.assert_allclose(energies[0], energies[1], atol=1e-5)

    def test_zero_slack_and_constant_cuts_compile_without_extra_slack(self):
        data = one_hour_data()
        zero = np.zeros((1, 3))
        beta = zero.copy()
        beta[0, 0] = -1.
        cuts = [BendersCut(alpha, zero, zero, zero, "feasibility") for alpha in (-1., 0., 1.)]
        cuts.append(BendersCut(1., beta, zero, zero, "feasibility"))
        master = QuantumMaster(data, [], cuts, theta_cap=100.)
        self.assertTrue(np.all(np.isfinite(master.matrix())))
        self.assertFalse(any(name.startswith("_slack_cut_") for name in master.qubo_variables()))
        values = {name: 0 for name in master.qubo_variables()}
        constant = master.model.hard_constraints_made["cut_2"].constraint_expr
        self.assertGreater(constant.get_val(values), 0.)
        zero_range = master.model.hard_constraints_made["cut_3"].constraint_expr
        values["u_0_0"] = 1
        self.assertAlmostEqual(zero_range.get_val(values), 0.)

    def test_scenario_cuts_keep_a_feasible_dispatch_unpenalized(self):
        data = one_hour_data()
        scenario = Scenario("small", 1., data.base_load)
        good = make_candidate(data, [scenario], np.array([[1, 0, 0]]))
        bad = make_candidate(data, [scenario], np.zeros((1, 3), dtype=int))
        cuts = []
        for candidate in (good, bad):
            result = solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w)
            cuts.append(BendersCut(
                result.cut_alpha, result.cut_beta_u, result.cut_beta_v, result.cut_beta_w,
                "optimality" if result.feasible else "feasibility",
            ))
        master = QuantumMaster(data, [scenario], cuts, theta_cap=100., theta_step=.5)
        master.matrix()
        values = fixed_bits(master, good)
        # This dispatch costs .01 * 30^2 + 2 * 30 = 69, exactly 138 theta steps.
        values.update({f"theta_{k}": (138 >> k) & 1 for k in range(master.theta_bits)})
        for index in range(2):
            encoded = master._encoded_inequalities[f"cut_{index}"]
            target = -encoded.constant - sum(value * values[key] for key, value in encoded.coefficients.items())
            self.assertGreaterEqual(target, 0)
            maximum = -encoded.constant - sum(min(0, value) for value in encoded.coefficients.values())
            slack = master.kw.core.Integer(f"_slack_cut_{index}", 0, maximum)
            for (name,), weight in sorted(slack.coefficient.items(), key=lambda item: -item[1]):
                values[name] = int(target >= weight)
                target -= values[name] * weight
            self.assertEqual(target, 0)
            expression = master.model.hard_constraints_made[f"cut_{index}"].constraint_expr
            self.assertAlmostEqual(expression.get_val(values), 0., delta=1e-9)
        encoded = master._encoded_inequalities["cut_1"]
        bad_values = fixed_bits(master, bad)
        self.assertGreater(encoded.constant + sum(
            value * bad_values[key] for key, value in encoded.coefficients.items()
        ), 0)

    def test_real_cut_keeps_matrix_scale_bounded(self):
        data, scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
        candidate = make_candidate(data, scenarios, np.ones((data.n_units, data.horizon), dtype=int))
        results = [solve_scenario(data, s, candidate.u, candidate.v, candidate.w) for s in scenarios]
        result = aggregate_optimality_cut(results, scenarios)
        cut = BendersCut(result.cut_alpha, result.cut_beta_u, result.cut_beta_v, result.cut_beta_w)
        before = QuantumMaster(data, scenarios, [], theta_cap=1e6)
        after = QuantumMaster(data, scenarios, [cut], theta_cap=1e6)
        self.assertLess(np.max(np.abs(after.matrix())), 100 * np.max(np.abs(before.matrix())))
        self.assertLess(len(after.qubo_variables()) - len(before.qubo_variables()), 40)
        self.assertLessEqual(after._encoded_inequalities["cut_0"].rounding_error, after.theta_step)
        self.assertIs(after.cuts[0], cut)


if __name__ == "__main__":
    unittest.main()
