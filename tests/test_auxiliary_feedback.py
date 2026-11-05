"""Cache/cut lifecycle and conditional auxiliary minimization regressions."""
from itertools import product
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import dimod
import numpy as np

from suc_benders.anneal import anneal
from suc_benders.benders import dispatch
from suc_benders.config import load_options
from suc_benders.qubo import bit_weights, encode_integer, polish_auxiliaries, qubo_energy, search_qubo
from suc_benders.search import _cache_key
from suc_benders.solver import _round_start, _round_progress
from suc_benders.validation import pack
from test_full_day_envelope import cut, minimum_by_original_bits, qubo_fixture, whole_fixture


class CacheLifecycleTests(unittest.TestCase):
    def test_evicted_feasible_cut_is_regenerated_without_counting_a_new_candidate(self):
        w = whole_fixture()
        best = pack(w.d, np.ones((2, 3)))
        other = pack(w.d, np.array([[0, 1, 1], [1, 1, 1]]))
        with patch('suc_benders.search.CUT_LIMIT', 2):
            self.assertTrue(w.accept(best, 'initial'))
            protected = w.best.cut
            self.assertTrue(w.accept(other, 'sa_raw'))
            self.assertIs(w.best.cut, protected)
            old = w.cuts[-1]
            w.add_cut(cut(0, np.zeros(len(best))))
            self.assertNotIn(_cache_key(other), w.cache)
            self.assertTrue(any(c is protected for c in w.cuts))
            self.assertFalse(any(c is old for c in w.cuts))
            run = NS(whole=w, result=w.result)
            start = _round_start(run)
            with patch('suc_benders.search.dispatch', wraps=dispatch) as called:
                self.assertTrue(w.accept(other, 'sa_raw'))
                self.assertTrue(w.accept(other, 'sa_raw'))
                self.assertEqual(called.call_count, 1)
            self.assertEqual(_round_progress(run, start)['new_candidates'], 0)
            self.assertEqual(w.search_statistics()['new_feasible'], 1)
            self.assertEqual(w.diagnostics['re_evaluations'], 1)
            self.assertEqual(w.diagnostics['cache_invalidations'], 1)
            self.assertEqual(len(w.diagnostics['evaluations']), 3)
            self.assertTrue(any(w.cut_cache_keys[id(c)] == _cache_key(other) for c in w.cuts))

    def test_feasibility_cut_invalidation_and_no_cut_cache(self):
        w = whole_fixture()
        best = pack(w.d, np.ones((2, 3)))
        off = pack(w.d, np.zeros((2, 3)))
        invalid = np.zeros(len(best))  # Missing shutdown at hour zero violates the binary transition.
        with patch('suc_benders.search.CUT_LIMIT', 2):
            w.accept(best, 'initial')
            self.assertFalse(w.accept(off, 'sa_raw'))
            self.assertFalse(w.accept(invalid, 'sa_raw'))
            w.add_cut(cut(0, np.zeros(len(best))))
            self.assertNotIn(_cache_key(off), w.cache)
            self.assertIn(_cache_key(invalid), w.cache)
            with patch('suc_benders.search.dispatch', wraps=dispatch) as called:
                self.assertFalse(w.accept(invalid, 'sa_raw'))
                self.assertFalse(w.accept(off, 'sa_raw'))
                self.assertEqual(called.call_count, 1)
            self.assertEqual(w.cut_totals['feasibility'], 2)

    def test_limit_one_protects_new_incumbent_and_does_not_revive_evicted_cache(self):
        w = whole_fixture()
        worse = pack(w.d, np.array([[0, 1, 1], [1, 1, 1]]))
        better = pack(w.d, np.ones((2, 3)))
        with patch('suc_benders.search.CUT_LIMIT', 1):
            w.accept(worse, 'initial')
            first_cut, first_cost = w.best.cut, w.result.objective
            w.accept(better, 'sa_raw')
            self.assertLess(w.result.objective, first_cost)
            self.assertIs(w.cuts[0], w.best.cut)
            self.assertIsNot(w.best.cut, first_cut)
            self.assertNotIn(_cache_key(worse), w.cache)
            w.accept(worse, 'sa_raw')
            self.assertEqual(len(w.cuts), 1)
            self.assertIs(w.cuts[0], w.best.cut)
            self.assertNotIn(_cache_key(worse), w.cache)
            with patch('suc_benders.search.dispatch', wraps=dispatch) as called:
                self.assertTrue(w.accept(better, 'sa_raw'))
                called.assert_not_called()


class AuxiliaryTests(unittest.TestCase):
    def check_conditional_minimum(self, q):
        minima = minimum_by_original_bits(q)
        rng = np.random.default_rng(7)
        for index, minimum in enumerate(minima):
            for _ in range(3):
                raw = rng.integers(0, 2, q.Q.shape[0])
                raw[:len(q.free)] = [(index >> j) & 1 for j in range(len(q.free))]
                corrected = polish_auxiliaries(q, raw)
                np.testing.assert_array_equal(corrected[:len(q.free)], raw[:len(q.free)])
                np.testing.assert_array_equal(polish_auxiliaries(q, corrected), corrected)
                self.assertAlmostEqual(qubo_energy(q, corrected), minimum, places=7)
                self.assertLessEqual(qubo_energy(q, corrected), qubo_energy(q, raw)+1e-8)
                self.assertAlmostEqual(qubo_energy(q, raw), raw@q.Q@raw+q.offset, places=7)

    def test_remainder_bit_encoding_and_input_validation(self):
        for maximum in (0, 1, 2, 3, 6, 7, 8, 13):
            weights = bit_weights(maximum)
            for value in range(maximum+1):
                self.assertEqual(encode_integer(value, weights)@weights, value)
        for value in (-1, .5, 8, np.nan):
            with self.assertRaises(ValueError):
                encode_integer(value, bit_weights(7))
        m, b = qubo_fixture(2)
        q = search_qubo(m, b, [], np.zeros(2), [], load_options())
        with self.assertRaises(ValueError):
            polish_auxiliaries(q, [.5, 0])

    def test_envelope_and_hard_slacks_reach_exhaustive_minimum(self):
        m, b = qubo_fixture(3, [1, 1, 1])
        cuts = [cut(0, [0, 0, 0]), cut(-.004, [.009, .008, -.003])]
        boundary = [(np.arange(3), np.array([1., 1., -1.]), 1., False)]
        q = search_qubo(m, b, cuts, np.zeros(3), boundary, load_options())
        self.assertEqual({r.kind for r in q.slacks}, {'hard', 'optimality'})
        self.check_conditional_minimum(q)

    def test_feasibility_relaxation_has_no_false_penalty_and_is_bounded(self):
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                a = np.array([.13, 1., .37, -.23]) if fixed else np.array([1., .37, -.23])
                c = -.54 if fixed else -.41
                m, b = qubo_fixture(len(a), fixed=[1., np.nan, np.nan, np.nan] if fixed else None)
                anchor = np.array([1., 0, 0, 0]) if fixed else np.zeros(3)
                q = search_qubo(m, b, [cut(c, a, False)], anchor, [], load_options())
                self.check_conditional_minimum(q)
                row = q.slacks[0]
                self.assertEqual(row.kind, 'feasibility')
                self.assertEqual(q.feasibility_relaxation_bound, 1/64)
                for bits in product((0, 1), repeat=3):
                    x = np.r_[1., bits] if fixed else np.array(bits)
                    true_residual = c+a@x
                    quantized = row.step*(row.constant+row.coefficients@bits)
                    self.assertLessEqual(quantized, true_residual+1e-12)
                    self.assertLess(true_residual-quantized, 1/64+1e-12)
                    sample = np.zeros(q.Q.shape[0]); sample[:3] = bits
                    if true_residual <= 0:
                        self.assertLess(abs(qubo_energy(q, polish_auxiliaries(q, sample))), 1e-20)

    def test_capacity_noninteger_demand_has_equivalent_zero_penalty(self):
        for demand in (0., 5.1, 6., 6.1):
            m, b = qubo_fixture(3)
            b.times, b.u = [0], np.arange(3).reshape(3, 1)
            b.d.pmax, b.d.reserve_fraction = np.array([2., 4., 6.]), 0.
            b.scenarios = [NS(load=np.array([demand]))]
            q = search_qubo(m, b, [], np.ones(3), [], load_options())
            self.check_conditional_minimum(q)
            for bits in product((0, 1), repeat=3):
                sample = np.zeros(q.Q.shape[0]); sample[:3] = bits
                energy = qubo_energy(q, polish_auxiliaries(q, sample))
                self.assertEqual(energy < 1e-20, b.d.pmax@bits >= demand)

    def test_fixed_only_and_no_auxiliary_cases(self):
        for fixed in (None, [1., 0.]):
            m, b = qubo_fixture(2, [1, 2], fixed)
            q = search_qubo(m, b, [cut(0, [0, 0])], np.array([1., 0.]), [], load_options())
            self.check_conditional_minimum(q)


class SegmentedSamplingTests(unittest.TestCase):
    def test_segments_receive_corrected_states_and_return_paired_diagnostics(self):
        m, b = qubo_fixture(2, [1, 1])
        q = search_qubo(m, b, [cut(0, [0, 0]), cut(-.004, [.009, .008])], np.zeros(2), [], load_options())
        o = load_options({'sa_temperature': 1., 'sa_cutoff': .001, 'sa_cooling': .8, 'sa_sweeps': 1})
        states, seeds = [], []
        def sample(bqm, **kwargs):
            state, labels = kwargs['initial_states']
            np.testing.assert_array_equal(state, polish_auxiliaries(q, state))
            self.assertEqual(kwargs['initial_states_generator'], 'none')
            self.assertLessEqual(len(kwargs['beta_schedule']), 8)
            states.append(state.copy()); seeds.append(kwargs['seed'])
            raw = np.ones(len(state), np.int8); raw[:2] = 0
            return dimod.SampleSet.from_samples((raw[::-1], labels[::-1]), vartype=dimod.BINARY, energy=0)
        with patch('suc_benders.anneal.SimulatedAnnealingSampler.sample', side_effect=sample):
            candidates, stats = anneal(q, o, 10., [7])
        chain = stats['chains'][0]
        self.assertGreater(len(states), 1)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(chain['segments_completed'], len(states))
        self.assertGreater(chain['auxiliary_energy_gap'], 0)
        self.assertEqual(chain['returned_auxiliary_gap'], 0)
        self.assertEqual(chain['quantized_envelope_excess'], 0)
        self.assertFalse(chain['budget_truncated'])
        np.testing.assert_array_equal(candidates, [[0, 0]])
        raw = np.array(list(stats['raw_bits'][0]), int)
        corrected = np.array(list(stats['corrected_bits'][0]), int)
        self.assertAlmostEqual(chain['raw_energy'], qubo_energy(q, raw))
        self.assertAlmostEqual(chain['energy'], qubo_energy(q, corrected))
        self.assertAlmostEqual(chain['auxiliary_energy_gap'], chain['raw_energy']-chain['energy'])

    def test_expired_budget_stops_before_next_segment_and_next_chain(self):
        m, b = qubo_fixture(2, [1, 1])
        q = search_qubo(m, b, [], np.zeros(2), [], load_options())
        now = [0.]
        def sample(bqm, **kwargs):
            now[0] = 2.
            return dimod.SampleSet.from_samples(np.zeros(2, np.int8), vartype=dimod.BINARY, energy=0)
        with patch('suc_benders.anneal.perf_counter', side_effect=lambda: now[0]), \
             patch('suc_benders.anneal.SimulatedAnnealingSampler.sample', side_effect=sample) as called:
            candidates, stats = anneal(q, load_options(), 1., [7, 17])
        self.assertEqual(called.call_count, 1)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(stats['chains'][0]['budget_truncated'])
        self.assertEqual(stats['chains'][0]['segments_completed'], 1)

    def test_high_coefficient_crossing_cuts_have_zero_returned_auxiliary_gap(self):
        n = 256
        rng = np.random.default_rng(17)
        m, b = qubo_fixture(n, np.ones(n))
        cuts = [cut(0, np.zeros(n))]+[cut(-.25, rng.uniform(-1, 1, n)) for _ in range(7)]
        q = search_qubo(m, b, cuts, np.zeros(n), [], load_options())
        self.assertGreater(np.abs(q.Q.data).max(), 1e7)
        o = load_options({'sa_sweeps': 1, 'sa_cooling': .5, 'sa_cutoff': .5})
        candidates, stats = anneal(q, o, 10, [7, 17])
        self.assertEqual(len(candidates), 2)
        for text, chain in zip(stats['corrected_bits'], stats['chains']):
            x = np.array(list(text), int)
            expected = q.linear@x+q.linear_offset
            self.assertAlmostEqual(qubo_energy(q, x), expected, places=8)
            self.assertEqual(chain['returned_auxiliary_gap'], 0)
            self.assertEqual(chain['quantized_envelope_excess'], 0)
            self.assertEqual(chain['quantized_envelope_violation'], 0)
            self.assertGreater(chain['segments_completed'], 1)
            self.assertGreater(chain['start_temperature'], o.sa_temperature)


if __name__ == '__main__':
    unittest.main()
