from copy import deepcopy

import numpy as np
import pytest

import program_evaluation as evaluation


def test_instances_reproduce_the_supplement_files(tmp_path):
    assert evaluation.write_instances(tmp_path/'x') == evaluation.instances()
    source = dict(reserve_fraction=.05, scenarios=[dict(name='s', probability=1., load=[1.23456]+[200.]*22+[3.45678])])
    old = deepcopy(source)
    for sign in (-1, 1):
        v = evaluation.perturb(source, sign)
        assert source == old and v['scenarios'][0]['load'][0] == 1.23456 and v['scenarios'][0]['load'][-1] == 3.45678
        assert v['scenarios'][0]['load'][10] != 200.


def test_job_matrix_rotates_arm_order_and_can_drop_the_early_program():
    jobs = evaluation.jobs()
    assert len(jobs) == 5*4+6*3 and len({j['id'] for j in jobs}) == len(jobs)
    assert sum(j['seconds'] for j in jobs) == 17400 and evaluation.LIMITS['runs']-17400 >= 15*len(jobs)
    firsts = [j['arm'] for k, j in enumerate(jobs[:20]) if k % 4 == 0]
    assert firsts == ['SA+R', 'SA+L118', 'SA+L24r2', 'SA+L24', 'SA+R']
    assert all(j['arm'] != 'SA+L24r2' for j in evaluation.jobs(False)) and len(evaluation.jobs(False)) == 33
    assert {j['seconds'] for j in jobs if j['variant'] == 'base'} == {600.}


def fake(arm, objective, auc):
    return dict(status='TIME_LIMIT', feasible=True, initial=dict(objective=100., u=[[1]]), initial_u_sha256='same',
                cost_tolerance=1e-4, objective=objective, auc=auc, arm=arm)


def runs(final_cost=90., final_auc=3.):
    values = {'SA+R': (95., 5.), 'SA+L118': (93., 4.), 'SA+L24r2': (94., 4.5), 'SA+L24': (final_cost, final_auc)}
    return [dict(fake(j['arm'], *values[j['arm']]), instance=j['instance'], seed=j['seed']) for j in evaluation.jobs()]


def test_criteria_follow_the_registered_gates():
    s = evaluation.summarize(runs())
    assert s['T1_iteration']['status'] == 'ITERATION_HELPS' and s['T2_vs_R']['status'] == 'PASS'
    assert s['T3_vs_L118']['status'] == 'SUPPORTED' and s['T3_vs_L118']['pairs'] == 11
    s = evaluation.summarize(runs(93.5, 4.6))
    assert s['T1_iteration']['status'] == 'NOT_SUPPORTED' and s['T3_vs_L118']['status'] == 'NOT_SUPPORTED'
    assert s['T2_vs_R']['status'] == 'FAIL'  # 降幅 8% < 10%
    no_early = [r for r in runs() if r['arm'] != 'SA+L24r2']
    s = evaluation.summarize(no_early, include_early=False)
    assert s['T1_iteration']['status'] == 'NO_IMPROVEMENT_AFTER_ROUND_1' and s['T2_vs_R']['status'] == 'PASS'


def test_missing_or_mismatched_runs_are_incomplete():
    rows = runs()
    rows[0]['initial_u_sha256'] = 'other'
    s = evaluation.summarize(rows)
    assert s['T2_vs_R']['status'] == 'INCOMPLETE'
    s = evaluation.summarize([r for r in runs() if not (r['arm'] == 'SA+L118' and r['seed'] == 83)])
    assert s['T3_vs_L118']['status'] == 'INCOMPLETE'
    assert evaluation.pair_result(fake('SA+R', 95., None), fake('SA+L24', 90., 2.))['auc_difference'] is None


def test_score_does_not_count_initialization_as_search():
    row = dict(initial=dict(objective=100., seconds=10., u=[[1]]), trajectory=[(2., 110.), (8., 100.), (20., 90.)],
               seconds=60., objective=90., feasible=True)
    r = evaluation.score(row, dict(status='OPTIMAL', objective=80., validation=dict(feasible=True)))
    assert r['auc'] == pytest.approx((20*25+40*12.5)/60) and r['first_improvement_seconds'] == 20.
    assert evaluation.score(row, {})['auc'] is None
