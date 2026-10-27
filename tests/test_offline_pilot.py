import hashlib
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
    source = pilot.L118.read_text(encoding='utf-8')
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
    runs = [pilot.run_trial('tiny', 'SA+L118', 5, 30., reference, rounds=2, burst_calls=2, data=(d, scenarios), o=options)
            for _ in range(2)]
    assert [c for _, c in runs[0]['trajectory']] == [c for _, c in runs[1]['trajectory']]
    assert runs[0]['operator_calls'] == runs[1]['operator_calls'] and runs[0]['feasible']
    plain = pilot.run_trial('tiny', 'SA', 5, 30., reference, rounds=2, data=(d, scenarios), o=options)
    assert plain['operator_calls'] == 0 and plain['sa_calls'] > 0 and plain['feasible']
    with pytest.raises(ValueError):
        pilot.run_trial('tiny', 'RS', 5, 30., reference, rounds=1, data=(d, scenarios), o=options)


def test_l118_proposes_on_the_multi_scenario_ieee118_initial_schedule():
    from suc_benders import solver
    d, scenarios = paper_pilot.instance('ieee118s10')
    o = paper_pilot.options('ieee118s10', 17, 60.)
    run = solver._Run(o, perf_counter(), {}, perf_counter())
    solver._initialize(run, d, scenarios, None)
    whole = run.whole
    program = pilot.Program(pilot.L118.read_text(encoding='utf-8'), perf_counter()+30)
    try:
        candidates = program.propose(pilot.context(d, scenarios, whole.result), 11, perf_counter()+30)
    finally:
        program.close()
        solver._shutdown(run)
    assert 1 <= len(candidates) <= 4
    assert all(pilot.candidate_u(whole.result.u, c).shape == (54, 24) for c in candidates)


# 程序改名后内容不变：三个冻结程序的 SHA256 与生成和验证时登记的一致。
def test_programs_keep_their_registered_hashes():
    expected = dict(L118='df7674289f2bef92fd38457a549711c6523b6bb9d6477cbce33ae487bd447373',
                    L24='4b0d6318b08cec176746100bf0f7f29d48d3dffb1dac22dd5f09668a5581a92c',
                    L24r2='2d39d418a91e6a1e97add2dea3eb413d2eb37f81ad24925a6f0c76b5b5ab13cb')
    assert {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in pilot.PROGRAMS.values()} == expected
    assert pilot.ARMS == ('SA', 'SA+R', 'SA+L118', 'SA+L24r2', 'SA+L24')
