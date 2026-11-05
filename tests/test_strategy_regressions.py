"""Correctness cases found in the strategy refactor review."""
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy import sparse

from run_experiment import freeze_experiment, run_summary, solution_snapshot
from suc_benders.config import load_options
from suc_benders.constraints import cover_policy, LegacySquarePolicy
from suc_benders.model import model, partition
from suc_benders.qubo import polish_auxiliaries, qubo_energy, search_qubo
from suc_benders.search import BlockSearch, WholeDay, _cache_key
from suc_benders.solver import Result
from suc_benders.strategies import SearchStrategies, coverage_selector
from suc_benders.validation import pack
from test_full_day_envelope import cut, qubo_fixture, tiny_problem


class StrategyRegressionTests(unittest.TestCase):
    def test_capacity_covers_are_globally_valid_and_minimal(self):
        bits = [np.array(x) for x in product((0, 1), repeat=3)]
        for pmax in (np.array([5., 4., 3.]), np.array([2.5, 1.2, .8])):
            for required in (0., .1, float(pmax[0]+1), float(pmax.sum())):
                b = NS(obj=np.zeros(3), binary=np.arange(3), u=np.arange(3).reshape(3, 1),
                       times=[0], d=NS(pmax=pmax, reserve_fraction=0.),
                       scenarios=[NS(load=np.array([required]))])
                rows = cover_policy.rows(b, [], load_options())
                for candidate in bits:
                    if pmax @ candidate < required:
                        rows += cover_policy.observe(b, candidate, 'repair_infeasible')
                for row in rows:
                    if row.kind != 'capacity_cover':
                        continue
                    with self.subTest(pmax=pmax, required=required, indices=row.indices):
                        for feasible in bits:
                            if pmax @ feasible >= required:
                                self.assertLessEqual(row.values @ feasible[row.indices], row.rhs)
                        capacity = pmax[row.indices]
                        threshold = pmax.sum()-required
                        self.assertGreater(capacity.sum(), threshold)
                        for value in capacity:
                            self.assertLessEqual(capacity.sum()-value, threshold+1e-9)

    def test_no_good_expires_only_when_the_outside_commitment_changes(self):
        d, _ = tiny_problem()
        d.horizon = 6
        scenarios = [NS(name='a', probability=1., load=np.full(6, 8.))]
        m = model(d, scenarios)
        full = partition(m, 6)[0][0]
        proposals, calls = [], []

        def sampler(q, options, seconds, seeds):
            calls.append(q)
            return ([proposals[0].copy()] if len(calls) == 1 else []), {'chains': []}

        o = load_options({'verbose': False, 'sa_constraint_policy': 'cover'})
        whole = WholeDay(m, full, o, perf_counter()+30, Result(), perf_counter(),
                         strategies=SearchStrategies(constraint_policy=cover_policy, sampler=sampler))
        initial = np.ones((2, 6)); initial[1, 1] = 0
        self.assertTrue(whole.accept(pack(d, initial), 'initial'))
        block = BlockSearch(partition(m, 3)[0][0], o, 1, whole)
        proposals.append(whole.best.point[block.columns].copy())
        block.step(1)
        self.assertTrue(any(row.kind == 'nogood' for row in block.constraint_rows))
        block._observe_constraint(np.zeros(len(block.columns), np.uint8), 'discrete_infeasible')
        block.step(2)
        self.assertTrue(any(row.kind == 'nogood' for row in block.constraint_rows))
        covers = [row.signature() for row in block.constraint_rows if row.kind == 'capacity_cover']
        self.assertTrue(covers)
        changed = np.ones((2, 6)); changed[1, 3:5] = 0
        self.assertTrue(whole.accept(pack(d, changed), 'changed_outside'))
        before = whole.result.objective
        candidate = whole.best.point[whole.full.binary].copy()
        candidate[block.columns] = proposals[0]
        self.assertNotIn(_cache_key(candidate), whole.cache)
        block.step(3)
        self.assertFalse(any(row.kind == 'nogood' for row in block.constraint_rows))
        self.assertEqual(covers, [row.signature() for row in block.constraint_rows
                                  if row.kind == 'capacity_cover'])
        self.assertTrue(whole.accept(candidate, 'combined'))
        self.assertTrue(whole.result.validation['feasible'])
        self.assertLess(whole.result.objective, before)

    def test_dominant_penalties_use_normalized_costs_and_residual_spacing(self):
        for n in (2, 3):
            energies = []
            for scale in (.001, 1., 1e6):
                m, b = qubo_fixture(n)
                cuts = [cut(n*scale, np.full(n, -scale)), cut(-1.9, np.ones(n), False)]
                q = search_qubo(m, b, cuts, np.zeros(n), [], load_options(),
                                constraint_policy=cover_policy)
                states = []
                for bits in product((0, 1), repeat=n):
                    sample = np.zeros(q.Q.shape[0], np.uint8)
                    sample[:n] = bits
                    states.append((bits, qubo_energy(q, polish_auxiliaries(q, sample))))
                self.assertLessEqual(sum(min(states, key=lambda entry: entry[1])[0]), 1)
                self.assertGreater(q.penalty_weights['feasibility']['minimum_penalty'],
                                   q.local_objective_bound)
                energies.append([energy for _, energy in states])
            np.testing.assert_allclose(energies[0], energies[1], rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(energies[1], energies[2], rtol=1e-12, atol=1e-9)

    def test_all_constraint_kinds_dominate_the_encoded_objective_range(self):
        m, b = qubo_fixture(5, obj=[2, 1, 3, 2, 1])
        m.A = sparse.csr_matrix([[1., -1., 1., -1., 1.]])
        m.rhs, m.eq = np.array([0.]), np.array([True])
        b.times, b.u = [0], np.arange(5).reshape(5, 1)
        b.d.pmax, b.d.reserve_fraction = np.array([5., 4., 3., 2., 1.]), 0.
        b.scenarios = [NS(load=np.array([6.]))]
        cuts = [cut(4, [-1, -2, 0, 0, 1]), cut(0, [1, 0, 2, 1, 0]),
                cut(-1.9, [1, 1, 0, 0, 0], False)]
        rows = cover_policy.observe(b, np.array([1, 0, 1, 0, 0]), 'repeated')
        q = search_qubo(m, b, cuts, np.zeros(5), [], load_options(),
                        constraint_policy=cover_policy, extra_rows=rows)
        self.assertEqual(set(q.penalty_weights), {'hard', 'capacity_cover', 'nogood', 'feasibility'})
        for entry in q.penalty_weights.values():
            self.assertGreater(entry['minimum_penalty'], q.local_objective_bound)
        objective = []
        for bits in product((0, 1), repeat=5):
            sample = np.zeros(q.Q.shape[0], np.uint8); sample[:5] = bits
            sample = polish_auxiliaries(q, sample)
            objective.append(q.linear @ sample)
        self.assertLessEqual(max(objective)-min(objective), q.local_objective_bound+1e-12)

    def test_legacy_energies_and_auxiliaries_match_the_archived_fixture(self):
        # Frozen from archive/pre-refactor-20261105 (9aaf7cc), without requiring Git at test time.
        m, b = qubo_fixture(3, obj=[2, 1, 3])
        cuts = [cut(4, [-1, -2, 0]), cut(0, [1, 0, 2]), cut(-2, [1, 2, 1], False)]
        q = search_qubo(m, b, cuts, np.zeros(3), [], load_options())
        energies = [1.3333333333333333, 2.333333333333333, 1., 3.00390625,
                    1.6666666666666665, 2.6705729166666665, 2.337239583333333, 7.005208333333333]
        auxiliaries = ['111111111000000001010101010', '000000010000000000101010100',
                       '000000000000000000101010100', '000000000100000000000000000',
                       '000000010000000000101010100', '000000000100000000000000000',
                       '000000000100000000000000000', '000000000111111110000000000']
        self.assertEqual(q.Q.shape, (30, 30))
        for bits, energy, auxiliary in zip(product((0, 1), repeat=3), energies, auxiliaries):
            sample = np.zeros(30, np.uint8); sample[:3] = bits
            corrected = polish_auxiliaries(q, sample)
            self.assertAlmostEqual(qubo_energy(q, corrected), energy, places=12)
            self.assertEqual(''.join(str(int(v)) for v in corrected[3:]), auxiliary)

    def test_coverage_spends_slots_on_envelope_improvement(self):
        cuts = [cut(0, [0])]
        cuts += [cut(-1e6, [1000*i]) for i in range(1, 8)]
        cuts += [cut(-1, [2])]
        for point in (np.zeros(1), np.ones(1)):
            values = [c.alpha+c.beta @ point for c in cuts]
            chosen, _ = coverage_selector(cuts, values, np.array([0]), 8, 64, point=point)
            self.assertEqual(set(chosen), {0, 8})

    def test_custom_policy_is_called_through_the_search(self):
        class Policy(LegacySquarePolicy):
            name = 'recording'

            def __init__(self):
                self.calls = set()

            def rows(self, *args):
                self.calls.add('rows')
                return super().rows(*args)

            def observe(self, *args):
                self.calls.add('observe')
                return []

            def weight(self, *args):
                self.calls.add('weight')
                return super().weight(*args)

        d, scenarios = tiny_problem()
        m = model(d, scenarios); full = partition(m, 3)[0][0]
        policy = Policy()

        def sampler(q, options, seconds, seeds):
            return [np.zeros(q.nx, np.uint8)], {'chains': []}

        whole = WholeDay(m, full, load_options({'verbose': False}), perf_counter()+30,
                         Result(), perf_counter(), strategies=SearchStrategies(policy, sampler=sampler))
        self.assertTrue(whole.diagnostics['sa_backend'].endswith('.sampler'))
        self.assertIsNone(whole.diagnostics['sa_initialization'])
        whole.accept(pack(d, np.ones((2, 3))), 'initial')
        BlockSearch(partition(m, 1)[0][0], whole.o, 1, whole).step(1)
        self.assertEqual(policy.calls, {'rows', 'observe', 'weight'})

    def test_backend_identity_reaches_the_result_summary_and_manifest(self):
        d, scenarios = tiny_problem()
        m = model(d, scenarios); full = partition(m, 3)[0][0]

        def sampler(q, options, seconds, seeds):
            return [], {'chains': [], 'sa_backend': 'tests.exact', 'sa_initialization': None}

        sampler.sa_backend = 'tests.exact'
        strategies = SearchStrategies(sampler=sampler)
        o = load_options({'verbose': False})
        result = Result(reference={'elapsed_seconds': 0.})
        whole = WholeDay(m, full, o, perf_counter()+30, result, perf_counter(), strategies=strategies)
        self.assertEqual(whole.diagnostics['sa_backend'], 'tests.exact')
        self.assertIsNone(whole.diagnostics['sa_initialization'])
        whole.accept(pack(d, np.ones((2, 3))), 'initial')
        BlockSearch(partition(m, 1)[0][0], o, 1, whole).step(1)
        saved = solution_snapshot(result, 0)
        self.assertIn('penalty_weights', saved['blocks'][0])
        summary = run_summary(result, NS(recording_seconds=0.), {}, 0., 0., perf_counter())
        with TemporaryDirectory() as folder:
            freeze_experiment(Path(folder), o, strategies=strategies)
            manifest = json.loads((Path(folder)/'manifest.json').read_text(encoding='utf-8'))
        for payload in (saved, summary, manifest, result.diagnostics['sampling'][0]):
            self.assertEqual(payload['sa_backend'], 'tests.exact')
            self.assertIsNone(payload['sa_initialization'])
            self.assertIn('sampler', payload)


if __name__ == '__main__':
    unittest.main()
