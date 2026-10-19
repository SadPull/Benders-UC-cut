from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
import pytest

from conftest import whole_day
from run_suc import _serializable
from suc_benders import search, solver
from suc_benders.benders import master
from suc_benders.config import load_options
from suc_benders.model import model, partition
from suc_benders.qubo import fixed_binary


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
    assert whole.diagnostics['sa_calls'] == whole.diagnostics['block_sa_calls'] == 2
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


def checkpoint(whole):
    state = _serializable(asdict(whole.result))
    state.update(charged_elapsed_seconds=2., resume_cursor=dict(outer=1, round=2),
                 resume=dict(source='test', block_state_rebuilt=True))
    return state


def test_basic_checkpoint_preserves_candidate_and_spent_budgets(tiny, options):
    original = whole_day(tiny, options)
    original.diagnostics['sa_budget_seconds_used'] = 3.
    original.result.llm_calls = 2
    original.advice.state['consecutive_failures'] = 1
    state = checkpoint(original); saved = deepcopy(state)
    restored = whole_day(tiny, options)
    restored.restore(state)
    assert restored.result.validation['feasible']
    assert restored.result.objective == original.result.objective
    assert restored.diagnostics['sa_budget_seconds_used'] == 3.
    assert restored.result.llm_calls == 2
    assert restored.advice.state['consecutive_failures'] == 1
    assert state == saved


def test_solver_resume_does_not_cold_initialize(tiny, options, monkeypatch):
    original = whole_day(tiny, options)
    state = checkpoint(original)
    state['llm_calls'] = 3
    monkeypatch.setattr(search.WholeDay, 'initialize', lambda *a: pytest.fail('Cold initialization'))
    backend = search.StrengthenedCertification
    monkeypatch.setattr(backend, 'start', lambda self: None)
    monkeypatch.setattr(backend, 'close', lambda self: None)
    monkeypatch.setattr(backend, 'step', lambda self: self.update_bound(self.result.objective))
    options.sa_enabled = False
    result = solver.solve(*tiny(), options, resume_state=state)
    assert result.status == 'CONVERGED' and result.validation['feasible']
    assert result.objective == pytest.approx(state['objective']) and result.llm_calls == 3
    assert result.elapsed_seconds >= state['charged_elapsed_seconds']


@pytest.mark.parametrize('legacy', [
    {'joint_sa_calls': 1}, {'joint_batches': [{'status': 'COMPLETE'}]},
    {'sampling': [{'kind': 'joint'}]},
])
def test_joint_checkpoint_is_rejected_before_reusing_evidence(tiny, options, monkeypatch, legacy):
    whole = whole_day(tiny, options)
    state = checkpoint(whole); state['diagnostics'].update(legacy)
    monkeypatch.setattr(search, 'dispatch', lambda *a, **k: pytest.fail('Old candidate reused'))
    with pytest.raises(ValueError, match='Joint-SA checkpoints are unsupported'):
        whole.restore(state)


@pytest.mark.parametrize('name,value', [('enhanced_search', False), ('block_sa_period', 3)])
def test_removed_search_options_are_rejected(tmp_path, name, value):
    with pytest.raises(ValueError, match='Unknown option'):
        load_options({name: value}, config_path=tmp_path/'absent.json')
