from time import perf_counter, sleep

import numpy as np
import pytest

import offline_pilot as pilot
import paper_pilot
from offline_worker import check_source
from suc_benders.reference import solve_reference
from conftest import make_whole


@pytest.mark.parametrize('source', [
    'def propose(context, rng):\n    import os\n    return []',
    'def propose(context, rng):\n    return context.__class__',
    'def propose(context, rng):\n    return open("x")',
    'def propose(context, rng, extra):\n    return []'])
def test_sandbox_rejects_forbidden_programs(source):
    with pytest.raises(ValueError):
        check_source(source)


def test_recovered_l3_passes_the_sandbox_and_proposes_decodable_candidates(tiny, options):
    source = pilot.L3.read_text(encoding='utf-8')
    check_source(source)
    whole = make_whole(*tiny(3, 6), options)
    program = pilot.Program(source, perf_counter()+30)
    try:
        candidates = program.propose(pilot.context(whole.d, whole.scenarios, whole.result), 7, perf_counter()+30)
    finally:
        program.close()
    assert isinstance(candidates, list) and len(candidates) <= 4
    for candidate in candidates:
        assert pilot.candidate_u(whole.result.u, candidate).shape == whole.result.u.shape


def test_burst_stops_after_the_sa_round_budget(tiny, options):
    class Slow:
        empty, calls = 0, 0
        def step(self):
            self.calls += 1
            sleep(.02)
    operator = Slow()
    spent = pilot.burst(operator, .1)
    # 只在每次调用之前检查时间，超出量不超过一次调用的用时。
    assert .1 <= spent <= .1 + .05 and operator.calls >= 4


def test_trials_are_reproducible_with_fixed_rounds_and_bursts(tiny, options):
    d, scenarios = tiny(3, 6)
    reference = solve_reference(d, scenarios, 30)
    # 参考成本压低 100，使手写算例在初始化后不会提前达到目标，循环确实执行。
    reference = dict(status=reference['status'], objective=reference['objective']-100., validation=reference['validation'])
    runs = [pilot.run_trial('tiny', 'L+SA', 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
            for _ in range(2)]
    assert [c for _, c in runs[0]['trajectory']] == [c for _, c in runs[1]['trajectory']]
    assert runs[0]['operator_calls'] == runs[1]['operator_calls'] and runs[0]['feasible']
    plain = pilot.run_trial('tiny', 'A', 5, 30., reference, rounds=2, data=(d, scenarios), o=options)
    assert plain['operator_calls'] == 0 and plain['sa_calls'] > 0 and plain['feasible']
    with pytest.raises(ValueError):
        pilot.run_trial('tiny', 'RS', 5, 30., reference, rounds=1, data=(d, scenarios), o=options)


def test_l3_proposes_on_the_multi_scenario_ieee118_initial_schedule():
    from suc_benders import solver
    d, scenarios = paper_pilot.instance('ieee118s10')
    o = paper_pilot.options('ieee118s10', 17, 60.)
    run = solver._Run(o, perf_counter(), {}, perf_counter())
    solver._initialize(run, d, scenarios, None)
    whole = run.whole
    program = pilot.Program(pilot.L3.read_text(encoding='utf-8'), perf_counter()+30)
    try:
        candidates = program.propose(pilot.context(d, scenarios, whole.result), 11, perf_counter()+30)
    finally:
        program.close()
        solver._shutdown(run)
    assert 1 <= len(candidates) <= 4
    assert all(pilot.candidate_u(whole.result.u, c).shape == (54, 24) for c in candidates)
