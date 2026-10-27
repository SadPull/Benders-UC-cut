import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

import program_evaluation as evaluation
import program_training as lt
import offline_pilot
import run_program_training as runner
from conftest import make_whole
from suc_benders.data import load_data
from suc_benders.reference import solve_reference

ROOT = Path(__file__).resolve().parents[1]


def test_original_prompt_renders_verbatim_and_substitutions_are_unique():
    d, scenarios = load_data(ROOT/'data/units.json', ROOT/'data/stochastic.json')
    text = runner.original_instructions(d, len(scenarios))
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == runner.ORIGINAL_INSTRUCTIONS_SHA256
    d, scenarios = load_data(runner.SOURCE/'units.json', runner.SOURCE/'stochastic.json')
    for r in (1, 2, 7):
        text = runner.instructions(d, len(scenarios), r)
        assert '3 load scenarios' in text and 'scenario_loads[scenario][hour]' in text
        assert f'round {r} of at least 4 and at most 10' in text
        assert ('Program 1 must refine' in text) == (r > 1)
        for old, new in runner.SUBSTITUTIONS:
            assert new in text


def test_stop_rule_never_stops_before_four_rounds_and_always_by_ten():
    rng = np.random.default_rng(3)
    for _ in range(300):
        values = list(np.maximum.accumulate(rng.choice([0., .05, .2, 1.], size=10)+rng.random(10)*.01))
        values = [None if rng.random() < .1 and k == 0 else v for k, v in enumerate(values)]
        for r in range(1, 11):
            decision = lt.stop_decision(values[:r])
            if r < lt.MIN_ROUNDS:
                assert not decision['stop']
            if r == lt.MAX_ROUNDS:
                assert decision['stop'] and decision['reason'] == 'MAX_ROUNDS'


def test_stop_rule_sequences():
    rising = [1.+k for k in range(10)]
    assert [lt.stop_decision(rising[:r])['stop'] for r in range(1, 11)] == [False]*9+[True]
    flat = [None]*10
    assert [lt.stop_decision(flat[:r])['reason'] for r in range(1, 5)] == ['CONTINUE']*3+['CONVERGED']
    once = [1.]*10
    assert lt.stop_decision(once[:3])['reason'] == 'CONTINUE' and lt.stop_decision(once[:4])['reason'] == 'CONVERGED'
    late = [1., 1., 1., 2., 2.02, 2.03]
    assert [lt.stop_decision(late[:r])['reason'] for r in (4, 5, 6)] == ['CONTINUE', 'CONTINUE', 'CONVERGED']
    # 阈值：max(0.10, 3% |B|)。
    assert lt.improved(1., 1.1) and not lt.improved(1., 1.0999)
    assert lt.improved(5., 5.15) and not lt.improved(5., 5.149)
    assert lt.improved(None, 0.) and not lt.improved(None, None) and not lt.improved(1., None)


def test_training_variants_exclude_the_test_instances(tmp_path):
    evaluation.write_instances(tmp_path/'tests')
    tests = [lt.scenario_loads(load_data(tmp_path/'tests'/n/'units.json', tmp_path/'tests'/n/'stochastic.json')[1])
             for n in evaluation.instances()]
    units = json.loads((runner.SOURCE/'units.json').read_text())
    assert (0., 0.) not in lt.SPLITS['train']+lt.SPLITS['validation']
    assert list(lt.variants()) == ['train1', 'train2', 'train3', 'validation1', 'validation2']
    for spec in lt.variants().values():
        (tmp_path/'u.json').write_text(json.dumps(dict(units, base_load=lt.load_profile(units['base_load'], spec['a'], spec['b']).tolist())))
        d, scenarios = load_data(tmp_path/'u.json', runner.SOURCE/'stochastic.json')
        assert len(scenarios) == 3 and np.isclose(d.reserve_fraction, .05)
        assert min(lt.test_distance(lt.scenario_loads(scenarios), t) for t in tests) >= lt.MIN_TEST_DISTANCE
    base = lt.scenario_loads(load_data(runner.SOURCE/'units.json', runner.SOURCE/'stochastic.json')[1])
    assert lt.test_distance(base, tests[0]) == 0.


def test_anchors_are_initial_half_reduction_and_final():
    a = lambda t, c: (t, NS(objective=c))
    initial, path = a(1., 100.), [a(2., 96.), a(3., 94.), a(9., 92.), a(20., 90.)]
    chosen = lt.choose_anchors(initial, path)
    assert [chosen[k][0] for k in lt.ROLES] == [1., 3., 20.]
    assert all(v is initial for v in lt.choose_anchors(initial, []).values())
    assert lt.choose_anchors(initial, [a(5., 99.)])['half'][0] == 5.


def row(generator, stage, variant, anchor, j=1., status='TIME_LIMIT'):
    return dict(stage=stage, variant=variant, generator=generator, anchor=anchor, status=status, J=j, feasible=True,
                elapsed_seconds=10., budget_seconds=15, llm_calls=0, message='', folder=f'{stage}-{variant}-{anchor}-{generator}',
                statistics=dict(calls=1, rejections=dict(capacity=2, timeline=1)))


def rows_for(generator, train_j, validation_j):
    out = []
    for stage, value in (('train', train_j), ('validation', validation_j)):
        for v in [n for n in lt.variants() if n.startswith(stage)]:
            for a in lt.ROLES:
                out.append(row(generator, stage, v, a, value))
    return out


def test_feedback_covers_every_earlier_program_and_withholds_validation():
    rows = rows_for('R', .5, .4)+rows_for('P1a', 2., 9.)
    programs = dict(P1a=dict(round=1, status='EVALUATED', sha256='a', reason='x'),
                    P1b=dict(round=1, status='STATIC_INVALID', sha256='b', error='Forbidden name: open'),
                    P2a=dict(round=2, status='PENDING_REVIEW', sha256='c'))
    entries = lt.feedback(rows, programs, 2, lambda i: 'def propose(context, rng):\n    return []')
    names = [e['generator'] for e in entries]
    assert names == lt.CONTROLS+['P1a', 'P1b']
    p1a = entries[names.index('P1a')]
    assert p1a['train_mean_J'] == 2. and len(p1a['measurements']) == 9 and p1a['code']
    assert p1a['statistics']['rejections'] == dict(capacity=18, timeline=9)
    assert entries[names.index('P1b')]['error'] and entries[names.index('P1b')]['code']
    assert '9.0' not in json.dumps(entries)
    record = lt.closed_loop_check(2, entries, programs, rows)
    assert [r['program'] for r in record] == ['P1a', 'P1b'] and len(record[0]['train_rows']) == 9
    with pytest.raises(RuntimeError, match='omits'):
        lt.closed_loop_check(2, [e for e in entries if e['generator'] != 'P1b'], programs, rows)
    short = [r for r in rows if not (r['generator'] == 'P1a' and r['anchor'] == 'half' and r['variant'] == 'train1')]
    with pytest.raises(RuntimeError, match='lacks'):
        lt.closed_loop_check(2, lt.feedback(short, programs, 2, lambda i: ''), programs, short)
    approved = dict(programs, P1b=dict(round=1, status='APPROVED', sha256='b'))
    with pytest.raises(RuntimeError, match='no recorded reason'):
        lt.closed_loop_check(2, lt.feedback(rows, approved, 2, lambda i: ''), approved, rows)
    leaked = entries+[dict(generator='X', measurements=[dict(variant='validation1')])]
    with pytest.raises(RuntimeError, match='Validation'):
        lt.closed_loop_check(2, leaked, programs, rows)


def test_feedback_attaches_source_to_top_programs_and_the_previous_round():
    rows, programs = [], {}
    for k in range(6):
        i = f'P{k//2+1}{"ab"[k%2]}'
        programs[i] = dict(round=k//2+1, status='EVALUATED', sha256=i)
        rows += rows_for(i, float(k), 0.)
    programs['P4a'] = dict(round=4, status='NO_PROGRAM', sha256=None, error='format')
    entries = {e['generator']: e for e in lt.feedback(rows, programs, 5, lambda i: 'src '+i)}
    with_code = sorted(i for i, e in entries.items() if e.get('code'))
    assert with_code == ['P2a', 'P2b', 'P3a', 'P3b']
    assert entries['P4a']['code'] is None and entries['P4a']['error'] == 'format'


def test_selection_gate_and_best_validation():
    rows = rows_for('R', .5, .4)+rows_for('P1a', 2., 1.)+rows_for('P1b', 1., 1.5)+rows_for('P2a', 3., 1.5)
    programs = {i: dict(round=int(i[1]), status='EVALUATED') for i in ('P1a', 'P1b', 'P2a')}
    assert lt.best_validation(rows, programs, 1) == 1.5
    chosen = lt.select(rows, programs)
    assert chosen == dict(final='P1b', early='P1b')
    assert lt.gate(rows, 'P1b')['passed'] and not lt.gate(rows, 'R')['passed']
    rows[-1]['status'] = 'PROGRAM_INVALID'
    assert lt.mean_j(rows, 'P2a', 'validation', 6) is None


def test_scenario_fields_only_when_requested(tiny, options):
    d, scenarios = tiny(3, 6)
    scenarios = [NS(name='a', probability=.25, load=d.base_load*.9), NS(name='b', probability=.75, load=d.base_load)]
    plan = NS(u=np.ones((3, 6)), scenarios=[NS(p=np.full((3, 6), 9.)), NS(p=np.full((3, 6), 11.))])
    plain, extra = offline_pilot.context(d, scenarios, plan), offline_pilot.context(d, scenarios, plan, True)
    assert set(extra)-set(plain) == {'scenario_loads', 'scenario_probabilities'}
    assert {k: extra[k] for k in plain} == plain
    assert np.allclose(plain['load'], .25*.9*d.base_load+.75*d.base_load) and np.allclose(plain['p'], 10.5)
    assert extra['scenario_probabilities'] == [.25, .75] and np.allclose(extra['scenario_loads'][0], d.base_load*.9)


@pytest.mark.parametrize('identifier', ['H0', 'R', 'L118'])
def test_probe_operator_records_rejection_reasons(tiny, options, identifier):
    whole = make_whole(*tiny(3, 6), options)
    source = offline_pilot.L118.read_text(encoding='utf-8') if identifier == 'L118' else None
    operator = lt.Operator(whole, identifier, source, 1, 2)
    plan = lt.snapshot(whole.result)
    try:
        for _ in range(3):
            operator.step(plan)
    finally:
        operator.close()
    stats = lt.statistics(whole.diagnostics)
    assert stats['calls'] == 3 and set(stats['rejections']) == set(lt.REJECTIONS)
    assert stats['returned_candidates'] == stats['format_valid']


def v110():
    try:
        source = subprocess.run(['git', 'show', 'b2f9f4e:offline_pilot.py'], cwd=ROOT, capture_output=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('v1.1.0 source unavailable')
    return source


def test_fixed_round_sa_and_l118_match_v110(tiny, options, tmp_path):
    path = tmp_path/'offline_pilot_v110.py'
    path.write_bytes(v110())
    spec = importlib.util.spec_from_file_location('offline_pilot_v110', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    old.ROOT, old.L3 = offline_pilot.ROOT, offline_pilot.L118
    d, scenarios = tiny(3, 6)
    reference = solve_reference(d, scenarios, 30)
    reference = dict(status=reference['status'], objective=reference['objective']-100., validation=reference['validation'])
    # v1.1.0 的 A、L+SA 即现在的 SA、SA+L118。
    for old_arm, arm in (('A', 'SA'), ('L+SA', 'SA+L118')):
        a = old.run_trial('tiny', old_arm, 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
        b = offline_pilot.run_trial('tiny', arm, 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
        assert a['final_u'] == b['final_u'] and a['operator_calls'] == b['operator_calls']
        assert a['operator_outcomes'] == b['operator_outcomes'] and a['sa_calls'] == b['sa_calls']
        assert [c for _, c in a['trajectory']] == pytest.approx([c for _, c in b['trajectory']])
        assert a['objective'] == pytest.approx(b['objective']) and a['status'] == b['status']


def test_new_arms_run_their_programs_with_scenario_fields(tiny, options, tmp_path, monkeypatch):
    seen = {}
    original = offline_pilot.context

    def spy(d, scenarios, plan, scenario_fields=False):
        seen.setdefault('fields', set()).add(scenario_fields)
        return original(d, scenarios, plan, scenario_fields)
    monkeypatch.setattr(offline_pilot, 'context', spy)
    for name in ('L24', 'L24r2'):
        path = tmp_path/f'{name}.py'
        path.write_text(offline_pilot.L118.read_text(encoding='utf-8'), encoding='utf-8')
        monkeypatch.setitem(offline_pilot.PROGRAMS, f'SA+{name}', path)
    d, scenarios = tiny(3, 6)
    for arm in ('SA+L24', 'SA+L24r2'):
        row = offline_pilot.run_trial('tiny', arm, 5, 30., {}, rounds=1, burst_calls=1, data=(d, scenarios), o=options)
        assert row['operator_calls'] == 1 and row['feasible']
    assert seen['fields'] == {True}
    seen.clear()
    offline_pilot.run_trial('tiny', 'SA+L118', 5, 30., {}, rounds=1, burst_calls=1, data=(d, scenarios), o=options)
    assert seen['fields'] == {False}


def test_parse_programs_and_websocket_args():
    good = {'programs': [{'reason': 'a', 'source': 'def propose(context, rng):\n    return []'}]*2}
    assert len(runner.parse_programs(json.dumps(good))) == 2
    for bad in ({'programs': good['programs'][:1]},
                {'programs': [{'reason': 'a', 'source': 'def propose(ctx, rng):\n    return []'}]*2}):
        with pytest.raises(ValueError):
            runner.parse_programs(json.dumps(bad))
    args = ['codex', 'exec', '-c', 'model_provider="suc_https"', '-c', 'model_providers.suc_https={x}',
            '-c', 'model_reasoning_effort="max"', '-c', 'model_providers.other.supports_websockets=false', '-']
    assert runner.websocket_args(args) == ['codex', 'exec', '-c', 'model_reasoning_effort="max"', '-']


def test_capture_records_completion_and_enforces_the_deadline(tmp_path):
    import sys
    prompt = tmp_path/'prompt.txt'; prompt.write_text('x')
    ok = runner.capture([sys.executable, '-c', 'print(\'{"type":"turn.completed"}\')'], None, prompt, tmp_path, 30, verbose=False)
    assert ok['status'] == 'OK' and ok['turn_completed_seconds'] is not None
    started = perf_counter()
    slow = runner.capture([sys.executable, '-c', 'import time; time.sleep(30)'], None, prompt, tmp_path, 1, verbose=False)
    assert slow['status'] == 'TIME_LIMIT' and perf_counter()-started < 15
