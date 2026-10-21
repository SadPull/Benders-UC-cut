import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy import sparse

from conftest import whole_day
from suc_benders import search, solver
from suc_benders.benders import master
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.qubo import fixed_binary


@pytest.mark.parametrize('matrix,maximum', [
    (sparse.csr_matrix((0, 0)), 0.),
    (sparse.csr_matrix((2, 2)), 0.),
    (sparse.csr_matrix([[0., -3.], [0., 0.]]), 3.),
])
def test_progress_handles_empty_and_nonempty_qubo(tiny, options, monkeypatch, matrix, maximum):
    whole = whole_day(tiny, options)
    block = search.BlockSearch(whole.full, options, 1, 1, whole)
    observed = []
    whole.progress = lambda state, event: observed.append(
        dict(state.diagnostics['pending_qubo'])) if event == 'before_sampling' else None
    q = NS(Q=matrix, fixed=whole.best.point[whole.full.binary].copy(), free=np.array([], int))
    monkeypatch.setattr(search, 'master', lambda b, cuts, seconds: (master(b, cuts)[0], -1e6))
    monkeypatch.setattr(search, 'search_qubo', lambda *args: q)
    monkeypatch.setattr(search, 'anneal', lambda *args: ([], {'bits': matrix.shape[0]}))
    block.step([whole.best.point[whole.full.binary]], 1)
    assert observed == [dict(bits=matrix.shape[0], nonzeros=matrix.nnz,
                             max_coefficient=maximum)]
    assert len(block.history) == 1 and whole.result.validation['feasible']


def test_each_block_sample_uses_derived_chain_seeds_and_current_incumbent(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    block = search.BlockSearch(whole.full, options, 1, 1, whole)
    seeds = []
    monkeypatch.setattr(search, 'master', lambda b, cuts, seconds: (master(b, cuts)[0], -1e6))
    monkeypatch.setattr(search, 'anneal', lambda q, o, seconds, seed:
                        (seeds.append(seed) or [], {'bits': q.Q.shape[0]}))
    for inner in (1, 2):
        block.step([whole.best.point[whole.full.binary]], inner)
    assert [len(s) for s in seeds] == [options.sa_reads, options.sa_reads]
    assert len({seed for s in seeds for seed in s}) == 2 * options.sa_reads
    assert seeds[0] == [int(np.random.SeedSequence([options.seed, 1, k]).generate_state(1)[0])
                        for k in range(options.sa_reads)]
    assert whole.diagnostics['sa_calls'] == 2
    assert all(row['kind'] == 'block' and row['baseline_u'] == whole.result.u.tolist()
               for row in whole.diagnostics['sampling'])
    assert not any(key.startswith('joint_') for key in whole.diagnostics)


def test_repair_runs_for_each_proposal_but_dispatch_still_deduplicates(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    calls = []
    repair = search.repair
    monkeypatch.setattr(search, 'repair', lambda *args: (calls.append(True), repair(*args))[1])
    raw = whole.result.u.copy(); raw[0] = 0
    first, second = {}, {}
    x = whole.prepare(raw, 'sa', first)
    assert whole.accept(x, 'sa', first)
    before = len(whole.diagnostics['evaluations'])
    y = whole.prepare(raw, 'sa', second)
    assert whole.accept(y, 'sa', second)
    assert len(calls) == 2 and np.array_equal(x, y)
    assert len(whole.diagnostics['evaluations']) == before
    assert second['outcome'] == 'duplicate' and second['incumbent_decrease'] == 0


def test_numerical_dispatch_status_rejects_candidate_without_caching(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    dispatch = search.dispatch
    monkeypatch.setattr(search, 'dispatch', lambda *a, **k: NS(status='SUBOPTIMAL', feasible=False))
    raw = whole.result.u.copy(); raw[0] = 0
    record = {}
    x = whole.prepare(raw, 'sa', record)
    assert not whole.accept(x, 'sa', record)
    assert record['outcome'] == 'dispatch_failed' and whole.diagnostics['dispatch_failures'] == 1
    monkeypatch.setattr(search, 'dispatch', dispatch)
    assert whole.accept(x, 'sa', {})


def test_reachability_survives_without_joint_master(tiny, options):
    d, scenarios = tiny(1, 6)
    d.ramp_up[:] = 5.; scenarios[0].load[:] = 0.
    m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
    native, _ = master(full, [])
    before = native.A.copy()
    started = perf_counter()
    whole = search.WholeDay(m, full, options, started+30, solver.Result(), started)
    assert (whole.fixed[full.u] == 0).all()
    for b in partition(m, 3)[0]:
        search.BlockSearch(b, options, 1, 1, whole)
        block_master, _ = master(b, [])
        assert (fixed_binary(block_master)[b.u] == 0).all()
    assert (master(full, [])[0].A != before).nnz == 0


@pytest.mark.parametrize('name,value', [('enhanced_search', False), ('block_sa_period', 3), ('sa_proximity', .3)])
def test_removed_search_options_are_rejected(tmp_path, name, value):
    with pytest.raises(ValueError, match='Unknown option'):
        load_options({name: value}, config_path=tmp_path/'absent.json')


def test_kaiwu_configuration_is_removed(tmp_path):
    with pytest.raises(ValueError, match='Unknown option'):
        load_options({'kaiwu_user_id': 'unused'}, config_path=tmp_path/'absent.json')
    path = tmp_path/'config.json'
    path.write_text('{"kaiwu": {"user_id": "unused"}}', encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown local configuration section'):
        load_options(config_path=path)


def test_http_channel_requires_url_and_key_without_codex(tmp_path):
    options = dict(llm_enabled=True, llm_model='m', llm_agent_directory=str(tmp_path))
    with pytest.raises(ValueError, match='URL and API key'):
        load_options(options, config_path=tmp_path/'absent.json')
    assert load_options(dict(options, llm_url='http://x', llm_api_key='k'), config_path=tmp_path/'absent.json').llm_enabled


def test_initial_output_must_be_explicit(tmp_path):
    data = Path(__file__).resolve().parents[1]/'data'
    units = json.loads((data/'units.json').read_text(encoding='utf-8'))
    del units['initial_output']
    path = tmp_path/'units.json'
    path.write_text(json.dumps(units), encoding='utf-8')
    with pytest.raises(ValueError, match='initial_output is required'):
        load_data(path, data/'stochastic.json')
