from itertools import product
from types import SimpleNamespace as NS

import dimod
import numpy as np
import pytest
from scipy import sparse

from conftest import whole_day
from suc_benders import anneal as sa, search


@pytest.mark.parametrize('values', [
    [[2., -3., 0.], [-3., -1., 0.], [0., 0., 0.]],
    [[2., -1., 0.], [4., -2., 0.], [0., 3., 0.]],
    np.zeros((3, 3)), np.zeros((0, 0)),
])
def test_sparse_binary_model_matches_every_qubo_energy(values, monkeypatch):
    Q = sparse.csr_matrix(values)
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', lambda *a, **k: pytest.fail('Dense QUBO'))
    bqm = sa.binary_model(Q)
    assert list(bqm.variables) == list(range(Q.shape[0]))
    assert bqm.vartype is dimod.BINARY
    for bits in product((0, 1), repeat=Q.shape[0]):
        x = np.array(bits, dtype=np.int8)
        assert bqm.energy(dict(enumerate(bits))) == pytest.approx(float(x @ Q @ x))


def problem():
    return NS(Q=sparse.csr_matrix([[1., -2., 0.], [-2., 1., .5], [0., .5, -1.]]),
              fixed=np.array([1., np.nan, 0., np.nan]), free=np.array([1, 3]), nx=4, energy_unit=2.)


def test_sampler_uses_random_starts_and_returns_terminal_bits_without_descent(options, monkeypatch):
    q, calls = problem(), []
    options.sa_temperature, options.sa_cooling, options.sa_cutoff = 4., .5, 1.
    def sample(self, bqm, **kwargs):
        calls.append((bqm, kwargs))
        # Return columns out of order to exercise label-based recovery.
        return dimod.SampleSet.from_samples(([[1, 0, 1]], [2, 0, 1]), dimod.BINARY, energy=0.)
    monkeypatch.setattr(sa.SimulatedAnnealingSampler, 'sample', sample)
    seeds = [2**31-1, 2**31, 2**32-1]
    candidates, stats = sa.anneal(q, options, 30., seeds)
    assert len(calls) == 3
    for k, (bqm, args) in enumerate(calls):
        assert args['seed'] == seeds[k] % 2**31
        assert args['num_reads'] == 1 and args['initial_states_generator'] == 'random'
        assert 'initial_states' not in args
        assert args['num_sweeps_per_beta'] == options.sa_sweeps
        assert args['beta_schedule_type'] == 'custom' and args['beta_schedule'] == [.125, .25]
        assert not args['randomize_order'] and args['proposal_acceptance_criteria'] == 'Metropolis'
        assert bqm.get_linear(0) == pytest.approx(1.)
        assert bqm is calls[0][0]
    assert all(x.tolist() == [1, 0, 0, 1] for x in candidates)
    assert stats['raw_bits'] == ['011'] * 3
    assert [c['seed'] for c in stats['chains']] == seeds
    assert [c['sampler_seed'] for c in stats['chains']] == [2**31-1, 0, 2**31-1]
    # The returned 011 state admits an improving flip; it must still reach repair unchanged.
    x = np.array([0, 1, 1]); improved = np.array([1, 1, 1])
    assert improved @ q.Q @ improved < x @ q.Q @ x


def test_real_random_sampler_is_repeatable_and_reports_terminal_energy(options):
    q = problem()
    first, a = sa.anneal(q, options, 30., [17, 19])
    second, b = sa.anneal(q, options, 30., [17, 19])
    assert len(first) == len(second) == 2
    assert all(np.array_equal(x, y) for x, y in zip(first, second))
    assert a['chains'] == b['chains'] and a['raw_bits'] == b['raw_bits']
    for bits, chain in zip(a['raw_bits'], a['chains']):
        x = np.array(list(bits), dtype=np.int8)
        assert chain['energy'] == pytest.approx(float(x @ q.Q @ x))


@pytest.mark.parametrize('sample,labels', [
    ([[0, 1]], [0, 1]), ([[0, 1, 0]], [0, 1, 3]),
    ([[0, 1, 2]], [0, 1, 2]), ([[0, 1, 0], [1, 0, 1]], [0, 1, 2]),
])
def test_invalid_sampler_output_is_rejected(options, monkeypatch, sample, labels):
    response = NS(vartype=dimod.BINARY, variables=labels, record=NS(sample=np.array(sample)))
    monkeypatch.setattr(sa.SimulatedAnnealingSampler, 'sample', lambda *a, **k: response)
    with pytest.raises(RuntimeError, match='invalid binary'):
        sa.anneal(problem(), options, 30., [1])


def test_deadline_stops_before_next_chain(options, monkeypatch):
    now, calls = [0.], []
    monkeypatch.setattr(sa, 'perf_counter', lambda: now[0])
    def sample(self, bqm, **kwargs):
        calls.append(kwargs)
        now[0] = 2.
        return dimod.SampleSet.from_samples([[0, 1, 1]], dimod.BINARY, energy=0.)
    monkeypatch.setattr(sa.SimulatedAnnealingSampler, 'sample', sample)
    _, stats = sa.anneal(problem(), options, 1., [1, 2, 3])
    assert len(calls) == len(stats['chains']) == 1 and stats['seconds'] == 2.
    calls.clear()
    assert sa.anneal(problem(), options, 0., [1])[1]['candidates'] == 0
    assert not calls


def test_model_build_time_is_charged(options, monkeypatch):
    now = [0.]
    original = sa.binary_model
    def build(Q):
        now[0] = 2.
        return original(Q)
    monkeypatch.setattr(sa, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(sa, 'binary_model', build)
    monkeypatch.setattr(sa.SimulatedAnnealingSampler, 'sample', lambda *a, **k: pytest.fail('Late chain'))
    assert sa.anneal(problem(), options, 1., [1])[1]['candidates'] == 0


def test_zero_bits_do_not_invoke_sampler(options, monkeypatch):
    q = NS(Q=sparse.csr_matrix((0, 0)), energy_unit=1.)
    monkeypatch.setattr(sa.SimulatedAnnealingSampler, 'sample', lambda *a, **k: pytest.fail('Empty sample'))
    candidates, stats = sa.anneal(q, options, 30., [17])
    assert candidates == [] and stats['bits'] == 0 and stats['chains'] == []


def test_real_sa_block_candidates_reach_whole_day_validation(tiny, options, monkeypatch):
    whole = whole_day(tiny, options, n=3)
    block = search.BlockSearch(whole.full, options, 1, whole)
    block.step([whole.best.point[whole.full.binary]], 1)
    sampling, = whole.diagnostics['sampling']
    assert sampling['sa_backend'] == sa.SA_BACKEND
    assert sampling['sa_initialization'] == sa.SA_INITIALIZATION
    assert len(sampling['chains']) == options.sa_reads
    assert sampling['raw_candidates'] == options.sa_reads
    assert any(row['source'] == 'sa_raw' for row in whole.diagnostics['local_evaluations'])
    assert any(row['proposal_id'].startswith('sa:') for row in whole.diagnostics['proposals'])
    assert whole.result.validation['feasible']
