import itertools
import unittest
from dataclasses import replace

import numpy as np

from suc_benders.candidates import make_candidate, reserve_capacity
from suc_benders.common import commitment_capacity, commitment_is_feasible
from suc_benders.context import _reserve_margin
from suc_benders.data import Scenario, UCData
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.subproblem import cut_value, solve_scenario


def one_hour_data():
    return UCData(
        n_units=1, horizon=3,
        pmax=np.array([55.]), pmin=np.array([10.]),
        pstart=np.array([45.]), pshut=np.array([35.]),
        min_up=np.array([1]), min_down=np.array([1]),
        ramp_up=np.array([135.]), ramp_down=np.array([135.]),
        initial_on=np.array([False]), initial_duration=np.array([1.]),
        initial_output=np.array([0.]), base_load=np.array([30., 0., 0.]),
        cost_a=np.array([.01]), cost_b=np.array([2.]), cost_c=np.array([5.]),
        startup_cost=np.array([3.]), shutdown_cost=np.array([4.]), reserve_fraction=0.,
    )


class CapacityTests(unittest.TestCase):
    def test_one_hour_startup_shutdown_capacity(self):
        data = one_hour_data()
        for demand, expected in ((30., True), (35., True), (35.1, False)):
            with self.subTest(demand=demand):
                scenario = Scenario("one_hour", 1., np.array([demand, 0., 0.]))
                candidate = make_candidate(data, [scenario], np.array([[1, 0, 0]]))
                np.testing.assert_allclose(commitment_capacity(
                    data, candidate.u, candidate.v, candidate.w), [[35., 0., 0.]])
                self.assertEqual(commitment_is_feasible(
                    data, [scenario], candidate.u, candidate.v, candidate.w), expected)
                margin = reserve_capacity(data, candidate, scenario)
                np.testing.assert_allclose(margin, [35. - demand, 0., 0.])
                np.testing.assert_allclose(_reserve_margin(data, [scenario], candidate), [margin])
                sub = solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w)
                self.assertEqual(sub.feasible, expected)
                full = solve_full(data, [scenario], FullConfig(output_flag=0))
                self.assertEqual(np.isfinite(full.objective), expected)
                if expected:
                    np.testing.assert_allclose(sub.p, [scenario.load], atol=1e-6)
                    np.testing.assert_array_equal(full.u, candidate.u)
                    self.assertAlmostEqual(full.expected_fuel_cost, sub.objective, places=5)
                else:
                    self.assertGreater(cut_value(sub, candidate.u, candidate.v, candidate.w), 0.)

    def test_split_capacity_cut_validity_on_all_small_commitments(self):
        original = one_hour_data()
        vectors = {key: np.repeat(value, 2) for key, value in vars(original).items()
                   if isinstance(value, np.ndarray) and key != "base_load"}
        data = replace(original, **vectors, n_units=2)
        scenario = Scenario("tiny", 1., data.base_load)
        records = []
        for bits in itertools.product((0, 1), repeat=6):
            candidate = make_candidate(data, [scenario], np.array(bits).reshape(2, 3))
            if np.any(candidate.w[:, -1]):
                continue
            result = solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w)
            records.append((candidate, result))
        feasible = [(c, r) for c, r in records if r.feasible]
        self.assertTrue(feasible)
        self.assertTrue(any(not r.feasible for _, r in records))
        for candidate, result in records:
            at_origin = cut_value(result, candidate.u, candidate.v, candidate.w)
            if result.feasible:
                self.assertAlmostEqual(at_origin, result.objective, places=5)
            else:
                self.assertGreater(at_origin, 1e-6)
            for other, dispatch in feasible:
                value = cut_value(result, other.u, other.v, other.w)
                self.assertLessEqual(value, (dispatch.objective if result.feasible else 0.) + 1e-5)


if __name__ == "__main__":
    unittest.main()
