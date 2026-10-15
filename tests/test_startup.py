from itertools import product
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from suc_benders import llm
from suc_benders.benders import dispatch
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import fixed_cost, startup_costs
from suc_benders.validation import pack, repair, unpack, validate


def oracle(d, u):
    """Count consecutive OFF entries backwards, independently of the model window."""
    costs = np.zeros_like(u, dtype=float)
    for i in range(d.n_units):
        prefix = ([1] if d.initial_on[i] else [1] + [0]*int(d.initial_duration[i]))
        for t in range(d.horizon):
            history = prefix + u[i, :t].tolist()
            if u[i, t] and not history[-1]:
                last_on = max(j for j, value in enumerate(history) if value)
                off = len(history)-last_on-1
                cold = off > d.min_down[i]+d.cold_start_hours[i]
                costs[i, t] = d.cold_startup_cost[i] if cold else d.startup_cost[i]
    return costs


@pytest.mark.parametrize("on,duration,cold_hours,schedule,expected", [
    (0, 1, 1, [1], 3.), (0, 2, 1, [1], 3.), (0, 3, 1, [1], 11.),
    (0, 1, 0, [1], 3.), (0, 2, 0, [1], 11.), (0, 100, 1, [1], 11.),
    (1, 1, 1, [0, 0, 1], 3.), (1, 1, 1, [0, 0, 0, 1], 11.),
    (1, 1, 1, [1, 1], 0.), (0, 1, 1, [0, 0, 0, 1], 11.),
    (0, 1, 1, [1, 0, 1, 0, 0, 0, 1], 17.),
])
def test_startup_boundary_and_full_dispatch(tiny, on, duration, cold_hours, schedule, expected):
    d, scenarios = tiny(1, len(schedule))
    d.initial_on[:] = on; d.initial_duration[:] = duration
    d.initial_output[:] = d.pmin*on
    d.cold_startup_cost = np.array([11.]); d.cold_start_hours = np.array([cold_hours])
    u = np.array([schedule], dtype=float)
    scenarios[0].load[:] = 30*u[0]
    np.testing.assert_array_equal(startup_costs(d, u), oracle(d, u))
    assert startup_costs(d, u).sum() == expected
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    r = dispatch(full, pack(d, u), 5, certified=True)
    assert r.feasible
    z = unpack(m, r.point)
    actual = expected + np.sum(d.cost_c[:, None]*u + d.shutdown_cost[:, None]*z.w)
    assert fixed_cost(d, u, z.w) == actual
    assert r.cost == pytest.approx(actual+69*u.sum(), abs=1e-5)
    assert validate(d, scenarios, z)["feasible"]
    if expected > 3*int(z.v.sum()):
        z.objective -= expected-3*z.v.sum()
        with pytest.raises(ValueError, match="cost validation"):
            validate(d, scenarios, z)


def test_cold_cost_changes_repair_choice(tiny):
    d, scenarios = tiny(2, 1)
    d.initial_duration[:] = [4, 1]
    d.cold_start_hours = np.ones(2)
    d.cold_startup_cost = np.full(2, 11.)
    repaired = repair(d, scenarios, np.zeros((2, 1)))
    np.testing.assert_array_equal(repaired[:2], [0, 1])


def test_cold_costs_and_all_cuts_against_exhaustive_schedules(tiny):
    d, _ = tiny(2, 4)
    d.pstart = d.pmax.copy(); d.pshut = d.pmax.copy()
    d.initial_duration[:] = 3
    d.cold_start_hours = np.zeros(2)
    d.cold_startup_cost = np.array([11., 8.])
    d.cost_b[1] = 2.7
    scenarios = [NS(name="low", probability=.4, load=np.array([30., 0., 0., 30.])),
                 NS(name="high", probability=.6, load=np.array([40., 0., 0., 45.]))]
    m = model(d, scenarios)
    assert len(m.binary) == 3*d.n_units*d.horizon
    full = partition(m, d.horizon)[0][0]
    feasible, cuts = [], []
    for bits in product([0., 1.], repeat=8):
        u = np.asarray(bits).reshape((2, 4))
        x = pack(d, u)
        r = dispatch(full, x, 5, certified=True)
        if r.cut is not None:
            cuts.append(r.cut)
        if not r.feasible:
            continue
        z = unpack(m, r.point)
        fuel = np.array([sum(s.probability*np.sum(d.cost_a*z.scenarios[k].p[:, t]**2
                         + d.cost_b*z.scenarios[k].p[:, t]) for k, s in enumerate(scenarios))
                         for t in range(d.horizon)])
        fees = oracle(d, u)
        premium = (fees-d.startup_cost[:, None]*z.v).sum(axis=0)
        assert r.cost == pytest.approx(fuel.sum()+fees.sum()
            + np.sum(d.cost_c[:, None]*u+d.shutdown_cost[:, None]*z.w), abs=1e-5)
        actual = np.array([float(full.obj[indices] @ r.point[indices]
                                  + full.q[indices] @ r.point[indices]**2)
                           for hour in full.times
                           for indices in [full.continuous[full.time[full.continuous] == hour]]])
        np.testing.assert_allclose(actual, fuel+premium, atol=1e-5)
        assert actual.sum() == pytest.approx(r.recourse, abs=1e-5)
        assert validate(d, scenarios, z)["feasible"]
        feasible.append((x, r, actual))
    assert len(feasible) > 1 and any(not c.optimality for c in cuts)
    for x, r, actual in feasible:
        for c in cuts:
            assert c.alpha+c.beta@x <= (r.recourse if c.optimality else 0.)+1e-7
    assert optimize(m, 5, integral=True).objval == pytest.approx(min(r.cost for _, r, _ in feasible))


def test_cold_cost_partition_identity(tiny):
    d, scenarios = tiny(1, 5)
    d.cold_startup_cost = np.array([11.]); d.cold_start_hours = np.array([2])
    m = model(d, scenarios)
    rng = np.random.default_rng(7)
    point = rng.random(len(m.obj)); prices = rng.random(len(m.rhs))
    blocks, cross = partition(m, 2, prices)
    assert any(m.labels[i] == "cold_start" for i in cross)
    value = sum(b.obj@point[b.cols]+b.q@(point[b.cols]**2)+b.constant for b in blocks)
    assert value == pytest.approx(m.obj@point+m.q@(point**2)+prices[cross]@(m.A[cross]@point-m.rhs[cross]))


def test_startup_details_reach_context_and_diagnostics(tiny, options):
    from conftest import whole_day

    def cold_tiny(n):
        d, scenarios = tiny(n)
        d.cold_startup_cost = np.full(n, 11.); d.cold_start_hours = np.zeros(n)
        d.initial_duration[:] = 3
        return d, scenarios

    whole = whole_day(cold_tiny, options, n=1)
    row = whole.diagnostics["evaluations"][0]
    assert row["fixed_cost"] == 26.
    assert row["expected_fuel_cost"] == pytest.approx(207.)
    assert sum(row["hourly_recourse_costs"]) == pytest.approx(215.)
    assert "hourly_fuel_costs" not in row
    prompt = json.loads(llm.request_body(whole.d, whole.scenarios, whole.result, options, None, [], 1)["input"])
    assert prompt["startup_cost"] == [3.]
    assert prompt["cold_startup_cost"] == [11.] and prompt["cold_start_hours"] == [0.]
    assert "min_down + cold_start_hours" in prompt["startup_rule"]
    from suc_benders.context import LLMContextInput, build_llm_context
    context = build_llm_context(whole.d, whole.scenarios, LLMContextInput(
        incumbent=whole.result, incumbent_results=whole.dispatches,
        base_evaluations=[], candidate_evaluations=[], history=[]))
    assert context["incumbent_cost"]["fixed"] == 26.
    assert context["incumbent_cost"]["expected_fuel"] == 207.
    assert context["unit_cost_decomposition"][0]["startup_cost"] == 11.
    assert context["hourly_commitment_profile"][0]["hourly_fixed_cost"] == 16.
    assert context["peaker_usage"][0]["incurred_startup_cost"] == 11.
    assert context["peaker_usage"][0]["cost_per_on_hour"] == pytest.approx(8.667)
