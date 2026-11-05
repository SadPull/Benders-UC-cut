"""Regression checks for full-day feedback, multicut QUBO and capacity equivalence.

Run: python -m unittest discover -s tests -p test_full_day_envelope.py -v
"""
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from suc_benders.anneal import anneal, binary_model
from suc_benders.benders import dispatch
from suc_benders.config import load_options, ROOT
from suc_benders.data import jsonable, load_data, save_json
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.qubo import bit_weights, search_qubo
from suc_benders.search import BlockSearch, SABudgetExhausted, WholeDay
from suc_benders.solver import Result, solve
from suc_benders.validation import pack


def cut(alpha, beta, optimality=True):
    return NS(alpha=float(alpha), beta=np.asarray(beta, float), optimality=optimality, certified=True)


def qubo_fixture(n=2, obj=None, fixed=None):
    rows = [] if fixed is None else np.flatnonzero(np.isfinite(fixed))
    m = NS(nx=n, A=sparse.eye(n, format='csr')[rows],
           rhs=np.asarray([] if fixed is None else np.asarray(fixed)[rows]), eq=np.ones(len(rows), bool))
    b = NS(obj=np.zeros(n) if obj is None else np.asarray(obj, float), binary=np.arange(n),
           u=np.empty((1, 0), int), times=[], d=NS(pmax=np.array([1.])))
    return m, b


def minimum_by_original_bits(q):
    # Exhaust every auxiliary assignment, independently of the encoder's bounds/penalty logic.
    n = q.Q.shape[0]
    assert n <= 20, n
    samples = ((np.arange(2 ** n, dtype=np.uint64)[:, None]
                >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
    energies = np.sum(samples * (q.Q @ samples.T).T, axis=1) + q.offset
    group = (samples[:, :len(q.free)] @ (2 ** np.arange(len(q.free)))).astype(int)
    minimum = np.full(2 ** len(q.free), np.inf)
    np.minimum.at(minimum, group, energies)
    return minimum


def tiny_problem():
    d = NS(n_units=2, horizon=3, pmax=np.array([10., 12.]), pmin=np.array([1., 1.]),
           pstart=np.array([10., 12.]), pshut=np.array([10., 12.]),
           min_up=np.ones(2), min_down=np.ones(2), initial_duration=np.ones(2),
           initial_on=np.ones(2), initial_output=np.array([5., 3.]),
           ramp_up=np.array([5., 12.]), ramp_down=np.array([5., 12.]),
           cost_a=np.array([.01, .02]), cost_b=np.array([1., 2.]), cost_c=np.array([3., 4.]),
           startup_cost=np.ones(2), shutdown_cost=np.zeros(2), reserve_fraction=.05)
    scenarios = [NS(name='a', probability=.5, load=np.array([8., 12., 8.])),
                 NS(name='b', probability=.5, load=np.array([7., 10., 7.]))]
    return d, scenarios


def whole_fixture():
    d, scenarios = tiny_problem()
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    whole = WholeDay(m, full, load_options({'verbose': False}), perf_counter()+60,
                     Result(), perf_counter())
    return whole


class EnvelopeTests(unittest.TestCase):
    def assert_envelope(self, cuts, obj=(1., 1.), fixed=None):
        m, b = qubo_fixture(len(obj), obj, fixed)
        anchor = np.zeros(len(obj)) if fixed is None else np.nan_to_num(fixed)
        q = search_qubo(m, b, cuts, anchor, [], load_options())
        minimum = minimum_by_original_bits(q)
        for index, energy in enumerate(minimum):
            x = q.fixed.copy()
            x[q.free] = [(index >> j) & 1 for j in range(len(q.free))]
            expected = (b.obj @ x + max([c.alpha+c.beta@x for c in cuts], default=0.))/q.cost_unit
            self.assertLessEqual(abs(energy-expected), q.envelope.error_bound + 1e-8)
        return q, minimum

    def test_new_nonanchor_cut_changes_minimized_search_cost(self):
        m, b = qubo_fixture(1)
        first, new = cut(0, [0]), cut(-1, [2])
        q1 = search_qubo(m, b, [first], np.zeros(1), [], load_options())
        q2 = search_qubo(m, b, [first, new], np.zeros(1), [], load_options())
        before = minimum_by_original_bits(q1)*q1.cost_unit
        after = minimum_by_original_bits(q2)*q2.cost_unit
        np.testing.assert_allclose(before, [0, 0], atol=1e-8)
        np.testing.assert_allclose(after, [0, 1], atol=1e-8)

    def test_crossing_cuts_intercepts_and_duplicates(self):
        base, a, b = cut(0, [0, 0]), cut(-.004, [.009, 0]), cut(-.003, [0, .008])
        q, values = self.assert_envelope([base, a, b])
        _, repeated = self.assert_envelope([base, a, b, deepcopy(a)])
        np.testing.assert_allclose(repeated, values, atol=1e-8)
        _, changed = self.assert_envelope([base, cut(-.008, [.009, 0]), b])
        self.assertGreater(values[1], changed[1])
        self.assertGreater(len(q.envelope.tau_ids), 0)

    def test_no_cuts_single_cut_and_fixed_variables(self):
        for cuts in ([], [cut(-3, [2, -1])]):
            with self.subTest(cuts=len(cuts)):
                q, _ = self.assert_envelope(cuts)
                self.assertEqual(len(q.envelope.tau_ids), 0)
        self.assert_envelope([cut(0, [0, 0]), cut(-.004, [.009, .008])], fixed=[1., np.nan])
        q, _ = self.assert_envelope([cut(0, [0, 0]), cut(-.004, [.009, .008])], fixed=[1., 0.])
        self.assertEqual(q.Q.shape, (0, 0))
        samples, stats = anneal(q, load_options(), 1., [7])
        np.testing.assert_array_equal(samples, [[1, 0]])
        self.assertEqual(stats['chains'], [])

    def test_integer_range_guard_and_invalid_point(self):
        with self.assertRaisesRegex(ValueError, 'exact double'):
            bit_weights(2 ** 53 + 1)
        m, b = qubo_fixture(1)
        with self.assertRaisesRegex(ValueError, 'exact double'):
            search_qubo(m, b, [cut(0, [0]), cut(-1e20, [1])], np.zeros(1), [], load_options())
        with self.assertRaisesRegex(ValueError, 'binary'):
            search_qubo(m, b, [], np.array([.5]), [], load_options())

    def test_dominated_working_cuts_do_not_claim_auxiliary_constraints(self):
        q, _ = self.assert_envelope([cut(0, [0, 0]), cut(-1, [0, 0])])
        self.assertEqual(q.envelope.cuts, 1)
        self.assertEqual(q.envelope.slack_bits, 0)
        self.assertEqual(len(q.envelope.tau_ids), 0)

    def test_capacity_and_feasibility_penalties_survive(self):
        m, b = qubo_fixture(1, [0])
        b.times, b.u = [0], np.array([[0]])
        b.d.reserve_fraction = 0.
        b.scenarios = [NS(load=np.array([1.]))]
        q = search_qubo(m, b, [], np.ones(1), [], load_options())
        np.testing.assert_allclose(minimum_by_original_bits(q), [4, 0])
        b.times, b.u = [], np.empty((1, 0), int)
        q = search_qubo(m, b, [cut(1, [-1], False)], np.ones(1), [], load_options())
        np.testing.assert_allclose(minimum_by_original_bits(q), [4, 0])

    def test_sampler_decoding_reproducibility_budget_and_diagnostics(self):
        q, _ = self.assert_envelope([cut(0, [0, 0]), cut(-.004, [.009, .008])])
        o = load_options({'sa_sweeps': 1, 'sa_temperature': 1., 'sa_cutoff': .5, 'sa_cooling': .5})
        first, stats = anneal(q, o, 5, [7, 17])
        second, repeated = anneal(q, o, 5, [7, 17])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(stats['raw_bits'], repeated['raw_bits'])
        bqm = binary_model(q.Q)
        for candidate, bits, corrected, chain in zip(first, stats['raw_bits'], stats['corrected_bits'], stats['chains']):
            bits = np.array(list(bits), dtype=int)
            corrected = np.array(list(corrected), dtype=int)
            np.testing.assert_array_equal(candidate[q.free], bits[:len(q.free)])
            self.assertAlmostEqual(bqm.energy(dict(enumerate(bits)))+q.offset, chain['raw_energy'], places=7)
            self.assertAlmostEqual(bqm.energy(dict(enumerate(corrected)))+q.offset, chain['energy'], places=7)
            required = max(0., (q.envelope.alpha+q.envelope.beta@bits[:len(q.free)]).max())
            self.assertAlmostEqual(chain['envelope_violation'], max(0., required-chain['tau']))
        self.assertEqual(anneal(q, o, 0, [7])[0], [])


class FeedbackTests(unittest.TestCase):
    def test_full_day_cut_detects_ramp_infeasibility_hidden_by_partition(self):
        d, _ = tiny_problem()
        for name, value in vars(d).copy().items():
            if isinstance(value, np.ndarray):
                setattr(d, name, value[:1].copy())
        d.n_units, d.horizon, d.reserve_fraction = 1, 2, 0.
        d.initial_output[:], d.ramp_up[:] = 2., 4.
        scenarios = [NS(name='ramp', probability=1., load=np.array([2., 8.]))]
        m = model(d, scenarios)
        x = pack(d, np.ones((1, 2)))
        full = partition(m, 2)[0][0]
        result = dispatch(full, x, 5, certified=True)
        self.assertFalse(result.feasible)
        self.assertFalse(result.cut.optimality)
        self.assertGreater(result.cut.alpha+result.cut.beta@x, 0.)
        for b in partition(m, 1)[0]:
            self.assertTrue(dispatch(b, x[b.cols[b.binary]], 5).feasible)

    def test_projection_changes_with_outside_incumbent_and_working_set(self):
        whole = whole_fixture()
        b = partition(whole.m, 1)[0][0]
        search = BlockSearch(b, whole.o, 1, whole)
        n = len(whole.full.binary)
        for i in range(12):
            whole.add_cut(cut(100 if i == 0 else i, np.arange(n) * .001))
        for i in range(70):
            whole.add_cut(cut(i, np.zeros(n), False))
        point = np.zeros(len(whole.m.obj))
        projected, counts = search.qubo_cuts(point)
        self.assertEqual(counts['optimality_cuts_selected'], 8)
        self.assertEqual(counts['feasibility_cuts_encoded'], 64)
        self.assertEqual([c.alpha for c in projected[:8]], [100, 11, 10, 9, 8, 7, 6, 5])
        self.assertEqual(projected[8].alpha, 69)
        original_alpha = whole.cuts[0].alpha
        outside = next(j for j in whole.full.binary if j not in search.columns)
        point[outside] = 1
        updated, _ = search.qubo_cuts(point)
        for bits in product((0, 1), repeat=len(search.columns)):
            candidate = point[whole.full.binary].copy(); candidate[search.columns] = bits
            self.assertAlmostEqual(updated[0].alpha+updated[0].beta@bits,
                                   whole.cuts[0].alpha+whole.cuts[0].beta@candidate)
        self.assertNotEqual(updated[0].alpha, projected[0].alpha)
        self.assertEqual(whole.cuts[0].alpha, original_alpha)

    def test_full_day_dispatch_cache_cuts_and_fifo(self):
        whole = whole_fixture()
        x = pack(whole.d, np.ones((2, 3)))
        with patch('suc_benders.search.dispatch', wraps=dispatch) as called:
            self.assertTrue(whole.accept(x, 'sa_raw'))
            self.assertTrue(whole.accept(x.copy(), 'sa'))
            self.assertEqual(called.call_count, 1)
            self.assertIs(called.call_args.args[0], whole.full)
            self.assertTrue(called.call_args.kwargs['certified'])
        self.assertTrue(whole.result.validation['feasible'])
        self.assertEqual(whole.cut_totals['optimality'], 1)
        before = whole.result.objective
        infeasible = pack(whole.d, np.zeros((2, 3)))
        self.assertFalse(whole.accept(infeasible, 'sa_raw'))
        self.assertEqual(whole.result.objective, before)
        self.assertEqual(whole.cut_totals['feasibility'], 1)
        search = BlockSearch(partition(whole.m, 1)[0][0], whole.o, 1, whole)
        self.assertEqual(search.qubo_cuts(whole.best.point)[1]['feasibility_cuts_encoded'], 1)
        with patch('suc_benders.search.CUT_LIMIT', 2):
            whole.add_cut(cut(0, np.zeros(len(x))))
        self.assertEqual(len(whole.cuts), 2)
        self.assertEqual(whole.diagnostics['cuts_dropped'], 1)

    def test_validation_failure_cannot_adopt_or_learn(self):
        whole = whole_fixture()
        with patch('suc_benders.search.validate', side_effect=ValueError('bad physical solution')):
            with self.assertRaises(ValueError):
                whole.accept(pack(whole.d, np.ones((2, 3))), 'sa_raw')
        self.assertIsNone(whole.best)
        self.assertEqual(whole.cuts, [])
        self.assertEqual(whole.cache, {})

    def test_batch_uses_frozen_baseline_even_after_an_improvement(self):
        whole = whole_fixture()
        whole.accept(pack(whole.d, np.ones((2, 3))), 'initial')
        block = BlockSearch(partition(whole.m, 1)[0][0], whole.o, 1, whole)
        baseline = whole.best.point.copy()
        x = baseline[block.columns].copy()
        outside = next(j for j in whole.full.binary if j not in block.columns)
        seen = []
        def accept(candidate, source, record):
            seen.append(candidate.copy())
            record['outcome'] = 'accepted'
            whole.best.point[outside] = 1 - baseline[outside]
            return True
        stats = {'chains': [], 'bits': len(x)}
        with patch('suc_benders.search.anneal', return_value=([x, x], stats)), patch.object(whole, 'accept', side_effect=accept):
            block.step(1)
        self.assertEqual(len(seen), 2)
        np.testing.assert_array_equal(seen[0], seen[1])
        self.assertEqual(seen[1][outside], baseline[outside])

    def test_quota_and_result_serialization(self):
        whole = whole_fixture()
        whole.o.sa_budget_seconds = 0
        with self.assertRaises(SABudgetExhausted):
            with whole.sa_work():
                self.fail('quota must stop before work')
        d, scenarios = tiny_problem()
        o = load_options({'verbose': False, 'block_hours': 1, 'max_rounds': 1,
                          'time_limit': 20, 'reference_time_limit': 2, 'sa_reads': 2, 'sa_sweeps': 1})
        # A missing reference exercises search rather than the optimal-reference early stop.
        reference = dict(status='TIME_LIMIT', objective=None, elapsed_seconds=0.)
        with patch('suc_benders.solver.solve_reference', return_value=reference):
            result = solve(d, scenarios, o)
        self.assertEqual(result.status, 'SEARCH_EXHAUSTED', result.message)
        self.assertTrue(result.validation['feasible'])
        self.assertEqual(result.diagnostics['complete_rounds'], 1)
        self.assertNotIn('local_evaluations', result.diagnostics)
        for row in result.diagnostics['sampling']:
            self.assertEqual(row['optimality_cuts_selected'],
                             row['optimality_cuts_encoded']+row['optimality_cuts_dominated'])
        with TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            save_json(path, jsonable(asdict(result)))
            payload = json.loads(path.read_text())
            self.assertEqual(payload['objective'], result.objective)
            self.assertIn('cost_unit', payload['diagnostics']['sampling'][0])


def with_legacy_capacity(m):
    """Independent pre-change capacity rows, including all duplicates."""
    keep = [i for i, label in enumerate(m.labels)
            if label not in ('startup_capacity', 'shutdown_capacity', 'envelope')]
    rows = []
    d, ix = m.d, m.ix
    for t in range(d.horizon):
        for i in range(d.n_units):
            for k in range(len(m.scenarios)):
                for which in range(3):
                    row = np.zeros(len(m.obj))
                    row[ix.p[i, t, k]], row[ix.r[i, t, k]], row[ix.u[i, t]] = 1, 1, -d.pmax[i]
                    if which == 0:
                        row[ix.v[i, t]] = d.pmax[i]-d.pstart[i]
                    elif which == 2 and t < d.horizon-1:
                        row[ix.w[i, t+1]] = d.pmax[i]-d.pshut[i]
                    rows.append(row)
    legacy = NS(**vars(m))
    legacy.A = sparse.vstack([m.A[keep], sparse.csr_matrix(rows)], format='csr')
    legacy.rhs = np.r_[m.rhs[keep], np.zeros(len(rows))]
    legacy.eq = np.r_[m.eq[keep], np.zeros(len(rows), bool)]
    legacy.labels = [m.labels[i] for i in keep]+['legacy_capacity']*len(rows)
    return legacy


class CapacityTests(unittest.TestCase):
    def test_default_and_rts_row_counts(self):
        for folder in ('data', 'instances/rts24'):
            d, scenarios = load_data(ROOT/folder/'units.json', ROOT/folder/'stochastic.json')
            m = model(d, scenarios)
            count = sum(label in ('envelope', 'startup_capacity', 'shutdown_capacity') for label in m.labels)
            expected = sum(max(1, int(d.pstart[i] < d.pmax[i])
                               + int(t < d.horizon-1 and d.pshut[i] < d.pmax[i]))
                           for i in range(d.n_units) for t in range(d.horizon))*len(scenarios)
            self.assertEqual(count, expected)
            if folder == 'data':
                self.assertEqual(count, 54*24)

    def test_dispatch_and_relaxation_equivalent_for_all_derating_cases(self):
        for start, stop in product((False, True), repeat=2):
            with self.subTest(start=start, stop=stop):
                d, scenarios = tiny_problem()
                if start:
                    d.pstart[0] = 4
                if stop:
                    d.pshut[0] = 3
                m = model(d, scenarios)
                legacy = with_legacy_capacity(m)
                full = partition(m, d.horizon)[0][0]
                old_full = partition(legacy, d.horizon)[0][0]
                relaxed, old_relaxed = optimize(m, 5), optimize(legacy, 5)
                self.assertEqual(relaxed.status, old_relaxed.status)
                self.assertAlmostEqual(relaxed.objval, old_relaxed.objval, places=5)
                for first in product((0, 1), repeat=d.horizon):
                    on = np.array([first, [1, 1, 1]], float)
                    x = pack(d, on)
                    new, old = dispatch(full, x, 5), dispatch(old_full, x, 5)
                    self.assertEqual(new.feasible, old.feasible)
                    if new.feasible:
                        self.assertAlmostEqual(new.cost, old.cost, places=5)


if __name__ == '__main__':
    unittest.main()
