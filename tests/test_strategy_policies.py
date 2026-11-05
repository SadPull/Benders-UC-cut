"""Coverage and strategy seams for the refactored block search."""
from types import SimpleNamespace as NS
import unittest
from time import perf_counter

import numpy as np
from scipy import sparse

from suc_benders.config import load_options
from suc_benders.constraints import cover_policy, legacy_square_policy
from suc_benders.qubo import search_qubo
from suc_benders.search import BlockSearch, WholeDay
from suc_benders.strategies import SearchStrategies, recent_tight_selector
from suc_benders.solver import Result
from suc_benders.validation import pack
from suc_benders.model import model, partition


def block_fixture():
    data = NS(pmax=np.array([6., 4., 3.]), reserve_fraction=0.)
    block = NS(obj=np.zeros(3), binary=np.arange(3), u=np.arange(3).reshape(3, 1),
               times=np.array([0]), d=data, scenarios=[NS(load=np.array([8.]))])
    model_view = NS(nx=3, A=sparse.csr_matrix((0, 3)), rhs=np.empty(0), eq=np.empty(0))
    return model_view, block


class StrategyPolicyTests(unittest.TestCase):
    def test_cover_policy_builds_a_zero_one_capacity_cover(self):
        _, block = block_fixture()
        rows = cover_policy.rows(block, [], load_options(), ())
        capacity = [row for row in rows if row.kind == 'capacity_cover']
        self.assertEqual(len(capacity), 1)
        np.testing.assert_array_equal(capacity[0].values, [-1.])
        self.assertEqual(capacity[0].rhs, -1.)

    def test_cover_observer_adds_a_no_good_for_an_invalid_assignment(self):
        _, block = block_fixture()
        rows = cover_policy.observe(block, np.zeros(3, dtype=np.uint8), 'discrete_infeasible')
        no_good = next(row for row in rows if row.kind == 'nogood')
        self.assertFalse(float(no_good.values @ np.zeros(3)) <= no_good.rhs)
        self.assertTrue(float(no_good.values @ np.array([1, 0, 0])) <= no_good.rhs)

    def test_search_qubo_reports_the_selected_policy_and_bound(self):
        m, block = block_fixture()
        q = search_qubo(m, block, [], np.zeros(3), [], load_options(), constraint_policy=cover_policy)
        self.assertEqual(q.constraint_policy, 'cover')
        self.assertGreaterEqual(q.constraint_rows, 1)
        self.assertGreaterEqual(q.local_objective_bound, 1.)

    def test_legacy_policy_keeps_the_archived_default(self):
        m, block = block_fixture()
        q = search_qubo(m, block, [], np.zeros(3), [], load_options())
        self.assertEqual(q.constraint_policy, 'legacy_square')

    def test_custom_selector_and_sampler_are_called(self):
        from test_full_day_envelope import tiny_problem

        data, scenarios = tiny_problem()
        full_model = model(data, scenarios)
        full = partition(full_model, data.horizon)[0][0]
        calls = {'selector': 0, 'sampler': 0}

        def selector(cuts, values, columns, optimality_limit, feasibility_limit):
            calls['selector'] += 1
            return recent_tight_selector(cuts, values, columns, optimality_limit, feasibility_limit)

        def sampler(q, options, seconds, seeds):
            calls['sampler'] += 1
            return [], {'chains': [], 'bits': q.Q.shape[0]}

        strategies = SearchStrategies(legacy_square_policy, selector, sampler)
        whole = WholeDay(full_model, full, load_options({'verbose': False}), perf_counter()+60,
                         Result(), perf_counter(), strategies=strategies)
        whole.accept(pack(data, np.ones((data.n_units, data.horizon))), 'initial')
        block = BlockSearch(partition(full_model, 1)[0][0], whole.o, 1, whole)
        block.step(1)
        self.assertEqual(calls['selector'], 1)
        self.assertEqual(calls['sampler'], 1)


if __name__ == '__main__':
    unittest.main()
