import json

import numpy as np
import pytest
from types import SimpleNamespace as NS

import make_ieee118_scenarios as generator
import offline_pilot
import paper_pilot as pilot
from conftest import make_whole
from run_paper_pilot import digest
from suc_benders.data import jsonable, load_data
from suc_benders.model import partition, priced
from suc_benders.reference import solve_reference
from suc_benders.validation import reachable_shortfall, transition_violations


def test_generator_is_reproducible_and_matches_the_instance(tmp_path):
    first, second = generator.files(), generator.files()
    assert first == second
    for name, content in first.items():
        assert (generator.TARGET/name).read_bytes() == content
        (tmp_path/name).write_bytes(content)
    assert digest(tmp_path/'units.json') == digest(generator.SOURCE/'units.json')
    d, scenarios = load_data(tmp_path/'units.json', tmp_path/'stochastic.json')
    assert len(scenarios) == generator.COUNT and abs(sum(s.probability for s in scenarios)-1) < 1e-9
    u = generator.max_commitment(d)
    assert not transition_violations(d, u)
    loads = np.array([s.load for s in scenarios])
    assert loads.max() <= generator.PEAK_SHARE*d.pmax.sum() and not reachable_shortfall(d, scenarios, u)


def test_context_is_unchanged_for_one_scenario_and_weighted_for_several(tiny, options):
    d, scenarios = tiny(2, 3)
    whole = make_whole(d, scenarios, options)
    old = jsonable(dict(n_units=d.n_units, horizon=d.horizon,
                        **{k: getattr(d, k) for k in offline_pilot.CONTEXT_FIELDS}, load=scenarios[0].load,
                        reserve_fraction=d.reserve_fraction, u=whole.result.u, p=whole.result.scenarios[0].p))
    assert json.dumps(offline_pilot.context(d, scenarios, whole.result)) == json.dumps(old)
    two = [NS(name='low', probability=.25, load=np.full(3, 30.)), NS(name='high', probability=.75, load=np.full(3, 40.))]
    whole = make_whole(d, two, options)
    c = offline_pilot.context(d, two, whole.result)
    assert c['load'] == pytest.approx([37.5]*3)
    p = .25*np.asarray(whole.result.scenarios[0].p) + .75*np.asarray(whole.result.scenarios[1].p)
    assert np.allclose(c['p'], p)


def test_recorder_matches_the_built_qubo(tiny, options):
    whole = make_whole(*tiny(3, 6), options)
    b = priced(partition(whole.m, 3)[0][0], np.zeros(len(partition(whole.m, 3)[0][0].crosseq)))
    search = pilot.CountingSearch(b, options, 1, whole)
    anchor = whole.best.point[b.cols[b.binary]]
    search.evaluate(anchor, 'initial')
    with pilot.counted([]) as log:
        q, _, encoded = search.build_qubo(anchor, search.boundary_rows())
    assert log[-1]['ours'] == q.Q.shape[0] and log[-1]['free'] == len(q.free) and log[-1]['master'] == 1
    assert log[-1]['feasibility_encoded'] == encoded and pilot.CountingSearch.log is None
    row = dict(free=100, master=5, stored_optimality=7, stored_feasibility=3)
    assert pilot.standard_bits(row, 1) == dict(same_cuts=100+12+10, single_cut=100+12+4)
    assert pilot.standard_bits(row, 12) == dict(same_cuts=100+12+120, single_cut=100+12+48)


@pytest.mark.parametrize('arm', ['A', 'L+SA'])
def test_recorder_does_not_change_the_trial(tiny, options, arm):
    d, scenarios = tiny(3, 6)
    reference = solve_reference(d, scenarios, 30)
    # 参考成本压低 100，使手写算例在初始化后不会提前达到目标，循环确实执行。
    reference = dict(status=reference['status'], objective=reference['objective']-100., validation=reference['validation'])
    plain = offline_pilot.run_trial('tiny', arm, 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
    counted = pilot.counted_trial('tiny', arm, 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
    assert [c for _, c in plain['trajectory']] == [c for _, c in counted['trajectory']]
    assert plain['final_u'] == counted['final_u'] and plain['objective'] == counted['objective']
    sizes = counted['qubo_sizes']
    assert sizes and all(s['ours'] >= s['free'] for s in sizes)
    for block in {s['block'] for s in sizes}:
        assert [s['master'] for s in sizes if s['block'] == block] == list(range(1, 1+sum(s['block'] == block for s in sizes)))


def run(seed, arm, gap, area, name='x'):
    return dict(instance=name, seed=seed, arm=arm, gap_percent=gap, auc=area)


def test_two_arm_evaluation_counts_wins_and_auc_reduction():
    seeds = (1, 2, 3, 4, 5)
    runs = [run(s, 'A', 1., 2.) for s in seeds] + [run(s, 'L+SA', .5, 1.) for s in seeds[:4]] + [run(5, 'L+SA', 1., 2.)]
    value = pilot.evaluate(runs, 'x', seeds)
    assert value == dict(pairs=5, wins=4, auc_reduction=.5, passed=True)
    assert not pilot.evaluate(runs[:-1], 'x', seeds)['passed']
    worse = [run(s, 'A', 1., 2.) for s in seeds] + [run(s, 'L+SA', .5, 1.9) for s in seeds]
    assert pilot.evaluate(worse, 'x', seeds)['auc_reduction'] == pytest.approx(.05) and not pilot.evaluate(worse, 'x', seeds)['passed']
    infeasible = [run(s, 'A', None, 2.) for s in seeds] + [run(s, 'L+SA', .5, 1.) for s in seeds]
    assert pilot.evaluate(infeasible, 'x', seeds)['wins'] == 5


def test_ruler_accepts_only_proven_references():
    valid = dict(feasible=True)
    assert pilot.ruler(dict(status='OPTIMAL', objective=1., validation=valid))['objective'] == 1.
    assert pilot.ruler(dict(status='TIME_LIMIT', objective=1., validation=valid, gurobi_gap_percent=.01))
    with pytest.raises(RuntimeError):
        pilot.ruler(dict(status='TIME_LIMIT', objective=1., validation=valid, gurobi_gap_percent=.2))
    with pytest.raises(RuntimeError):
        pilot.ruler(dict(status='OPTIMAL', objective=1., validation=dict(feasible=False)))
