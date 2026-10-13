import itertools
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import azema
from suc_benders.benders import dispatch, master
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.hourly_bounds import HourlyBounds
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import pack, repair, unpack, validate


@pytest.fixture
def sms(tiny):
    def make(n=1, T=3):
        d, scenarios = tiny(n, T)
        d.model_profile = "azema_sms"
        d.cost_a[:] = 0
        d.pstart = d.pmin.copy(); d.pshut = d.pmin.copy()
        d.ramp_up[:] = 15; d.ramp_down[:] = 10
        d.shedding_cost = d.curtailment_cost = 700.
        return d, scenarios
    return make


def test_sms_has_soft_balance_linear_cost_and_no_terminal_capacity_cuts(sms):
    d, scenarios = sms()
    m = model(d, scenarios)
    b = partition(m, d.horizon)[0][0]
    mp, _ = master(b, [])
    assert not m.q.any() and m.ix.r is None
    assert 'terminal' not in m.labels and 'reserve' not in m.labels
    assert 'capacity' not in mp.kind and not mp.and_pairs
    point = np.zeros(len(m.obj))
    point[m.ix.shed[:, 0]] = scenarios[0].load
    residual = m.A @ point-m.rhs
    residual[m.eq] = np.abs(residual[m.eq])
    assert residual.max() <= 1e-9
    result = unpack(m, point)
    audit = validate(d, scenarios, result)
    assert audit['feasible'] and not audit['zero_shedding']
    assert audit['expected_shed_mwh'] == 90
    assert result.objective == 63000 and result.scenarios[0].objective == 0


def test_sms_repair_does_not_force_capacity_or_forbid_terminal_shutdown(sms):
    d, scenarios = sms()
    assert not repair(d, scenarios, np.zeros((1, 3))).any()
    raw = np.array([[1, 1, 0]])
    np.testing.assert_array_equal(repair(d, scenarios, raw), pack(d, raw))


def test_sms_startup_and_shutdown_ramps_use_pmin(sms):
    d, scenarios = sms()
    m = model(d, scenarios)
    point = np.zeros(len(m.obj))
    point[m.binary] = pack(d, np.array([[1, 1, 0]]))
    point[m.ix.p[:, :, 0]] = [[10, 10, 0]]
    point[m.ix.shed[:, 0]] = [20, 20, 30]
    assert validate(d, scenarios, unpack(m, point))['feasible']
    point[m.ix.p[0, 0, 0]] = 11
    point[m.ix.shed[0, 0]] = 19
    with pytest.raises(ValueError, match='physical'):
        validate(d, scenarios, unpack(m, point))


def test_sms_initial_power_prevents_early_shutdown(sms):
    d, scenarios = sms()
    d.initial_on[:] = 1; d.initial_duration[:] = 10; d.initial_output[:] = 25
    np.testing.assert_array_equal(azema.repair(d, np.zeros((1, 3))), [[1, 1, 0]])
    assert list(azema.initial_forced_hours(d)) == [2]


def test_sms_loader_accepts_shortage_and_zero_duration_means_on(sms, tmp_path):
    d, _ = sms()
    raw = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(d).items()}
    raw.pop('initial_on'); raw['initial_duration'] = [0]; raw['initial_output'] = [10]
    units, scenarios = tmp_path/'units.json', tmp_path/'scenarios.json'
    units.write_text(json.dumps(raw))
    scenarios.write_text(json.dumps(dict(reserve_fraction=0, scenarios=[dict(name='shortage', probability=1, load=[100]*3)])))
    loaded, s = load_data(units, scenarios)
    assert loaded.initial_on[0] == 1 and loaded.initial_duration[0] == 0
    assert s[0].load.max() > loaded.pmax.sum()


def test_no_sa_option_requires_certification_and_no_llm(tmp_path):
    config = tmp_path/'absent.json'
    assert not load_options(dict(sa_enabled=False, bound_mode='integer_benders'), config).sa_enabled
    with pytest.raises(ValueError):
        load_options(dict(sa_enabled=False), config)
    with pytest.raises(ValueError):
        load_options(dict(sa_enabled=1), config)


def test_sms_prompts_report_correct_cost_and_terminal_rules(sms):
    from suc_benders.llm import request_body
    from suc_benders.bound_advisor import request_body as bound_body
    from suc_benders.context import LLMContextInput, build_llm_context
    from suc_benders.config import Options
    d, scenarios = sms()
    m = model(d, scenarios); point = np.zeros(len(m.obj)); point[m.ix.shed[:, 0]] = scenarios[0].load
    r = unpack(m, point)
    results = [NS(**vars(s), balance_dual=np.zeros(3), reserve_dual=np.zeros(3)) for s in r.scenarios]
    context = build_llm_context(d, scenarios, LLMContextInput(r, results, [], [], []))
    assert context['incumbent_cost']['total'] == r.objective
    assert context['incumbent_cost']['expected_penalty'] == 63000
    assert context['reserve_margin'] == []
    for body in (request_body(d, scenarios, r, Options(), context, [], 1),
                 bound_body(d, scenarios, r, Options(), {}, 1)):
        prompt = json.loads(body['input'])
        assert 'allowed' in prompt['terminal_rule'] and prompt['shedding_cost'] == 700


@pytest.fixture
def numerical_memory():
    import psutil
    import os
    if psutil.virtual_memory().available < 4*1024**3 and os.environ.get('SUC_EXPERIMENT_ALLOW_LOW_MEMORY') != '1':
        pytest.skip('4 GiB resource gate: no numerical solve started')


@pytest.mark.numerical
def test_sms_all_cuts_bound_every_feasible_commitment(sms, numerical_memory):
    d, scenarios = sms(1, 3)
    m = model(d, scenarios); b = partition(m, d.horizon)[0][0]
    observations, cuts = [], []
    hourly = HourlyBounds(b)
    for bits in itertools.product((0, 1), repeat=3):
        x = pack(d, np.array(bits)[None, :])
        r = dispatch(b, x, 10, certified=True)
        if r.cut is not None:
            cuts.append(r.cut)
        if r.feasible:
            assert validate(d, scenarios, unpack(m, r.point))['feasible']
            observations.append((x, r))
            list(hourly.observe(x, 'enumeration', hourly.recourse(r.point), lambda: 10))
    for x, r in observations:
        for cut in cuts:
            assert cut.alpha+cut.beta @ x <= (r.recourse if cut.optimality else 0)+1e-5
        assert hourly.lower(x) <= r.recourse+1e-5
    result = optimize(m, 20, integral=True)
    assert result.objval == pytest.approx(min(r.cost for _, r in observations), abs=1e-5)


@pytest.mark.numerical
def test_sms_certificate_only_never_samples_or_calls_llm(sms, numerical_memory, monkeypatch):
    from suc_benders import solver
    from suc_benders.config import Options
    d, scenarios = sms()
    monkeypatch.setattr(solver, 'anneal', lambda *a: pytest.fail('Unexpected SA call'))
    from suc_benders import bound_advisor
    monkeypatch.setattr(bound_advisor, 'BoundAdvisor', lambda *a: pytest.fail('Unexpected LLM call'))
    o = Options(sa_enabled=False, bound_mode='integer_benders', bound_slice_seconds=1,
                time_limit=30, tolerance=.001, verbose=False)
    result = solver.solve(d, scenarios, o)
    assert result.status == 'CONVERGED'
    assert result.diagnostics['sa_calls'] == result.llm_calls == 0
    assert result.validation['feasible'] and result.relative_gap <= .001+1e-8


@pytest.mark.numerical
def test_bounded_farkas_cut_separates_infeasible_recourse(numerical_memory):
    from scipy import sparse
    from audit_helpers import verify_cut
    b = NS(binary=np.array([0]), continuous=np.array([1]), A=sparse.csr_matrix([[2., -1.]]),
           rhs=np.array([0.]), eq=np.array([False]), obj=np.array([0., 1.]), q=np.zeros(2),
           lb=np.zeros(2), ub=np.ones(2), constant=0., audit_cuts=True)
    good = dispatch(b, np.array([0]), 10, certified=True)
    bad = dispatch(b, np.array([1]), 10, certified=True)
    assert good.feasible and not bad.feasible and not bad.cut.optimality
    verify_cut(b, vars(bad.cut))
    assert bad.cut.alpha+bad.cut.beta @ np.array([0]) <= 1e-5
    assert bad.cut.alpha+bad.cut.beta @ np.array([1]) > 1e-5
