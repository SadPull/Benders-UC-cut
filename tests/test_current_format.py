from copy import deepcopy
import json
from pathlib import Path

import pytest

from conftest import whole_day
from test_basic_search import checkpoint
from suc_benders.data import load_data


@pytest.mark.parametrize('key', ['history', 'outer_history'])
def test_missing_checkpoint_history_is_not_filled(tiny, options, key):
    whole = whole_day(tiny, options)
    state = checkpoint(whole); del state[key]
    with pytest.raises(KeyError, match=key):
        whole.restore(state)


@pytest.mark.parametrize('key', ['proposals', 'block_sa_calls', 'sa_budget_seconds_used',
                                 'llm_background_seconds'])
def test_missing_checkpoint_counters_are_not_reconstructed(tiny, options, key):
    whole = whole_day(tiny, options)
    state = checkpoint(whole); del state['diagnostics'][key]
    with pytest.raises(KeyError, match=key):
        whole.restore(state)


def test_unidentified_checkpoint_and_old_advisor_version_are_rejected(tiny, options):
    whole = whole_day(tiny, options)
    state = checkpoint(whole); del state['diagnostics']['search_mode']
    with pytest.raises(ValueError, match='current time-block format'):
        whole.restore(state)
    diagnostic = deepcopy(whole.diagnostics)
    diagnostic['advisor_state']['version'] = 1
    before = deepcopy(whole.advice.state)
    with pytest.raises(ValueError, match='current advice scheduling state'):
        whole.advice.restore(diagnostic)
    assert whole.advice.state == before


def test_incomplete_evaluation_is_not_silently_skipped(tiny, options):
    whole = whole_day(tiny, options)
    state = checkpoint(whole); del state['diagnostics']['evaluations'][0]['u']
    with pytest.raises(KeyError, match='u'):
        whole.restore(state)


def test_current_active_advice_checkpoint_preserves_time_and_spent_calls(tiny, options):
    original = whole_day(tiny, options)
    original.advice.begin()
    original.result.llm_calls = 1
    original.result.llm_history = [dict(status='RUNNING')]
    original.diagnostics['llm_background_seconds'] = 1.25
    restored = whole_day(tiny, options)
    restored.restore(checkpoint(original))
    assert restored.result.llm_calls == 1
    assert restored.result.llm_history[0]['status'] == 'CANCELLED'
    assert restored.diagnostics['llm_background_seconds'] == 1.25
    assert restored.advice.state['batch'] is restored.diagnostics['llm_batches'][0]
    assert restored.advice.worker is None


def test_initial_output_must_be_explicit(tmp_path):
    data = Path(__file__).resolve().parents[1]/'data'
    units = json.loads((data/'units.json').read_text(encoding='utf-8'))
    del units['initial_output']
    path = tmp_path/'units.json'
    path.write_text(json.dumps(units), encoding='utf-8')
    with pytest.raises(ValueError, match='initial_output is required'):
        load_data(path, data/'stochastic.json')

