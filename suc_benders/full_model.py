from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from .common import Scenario, require_gurobi
from .data import UCData


@dataclass
class FullConfig:
    time_limit: float | None = None
    mip_gap: float = 1e-6
    output_flag: int = 1


@dataclass
class ScenarioDispatch:
    name: str
    probability: float
    p: np.ndarray
    r: np.ndarray
    fuel_cost: float


@dataclass
class FullResult:
    status: str
    objective: float
    fixed_cost: float
    expected_fuel_cost: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    scenarios: list[ScenarioDispatch]
    mip_gap: float
    runtime: float
    model: Any = None
    lower_bound: float = float("-inf")


def solve_full(data: UCData, scenarios: Sequence[Scenario], config: FullConfig | None = None) -> FullResult:
    config = config or FullConfig()
    started = time.perf_counter()
    gp = require_gurobi()
    n, horizon = data.n_units, data.horizon
    model = gp.Model("suc_full")
    model.Params.OutputFlag = config.output_flag
    model.Params.MIPGap = config.mip_gap
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit

    u = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="u")
    v = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="v")
    w = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="w")
    scenario_vars = []
    for scenario in scenarios:
        p = model.addMVar((n, horizon), lb=0.0, name=f"p_{scenario.name}")
        r = model.addMVar((n, horizon), lb=0.0, name=f"r_{scenario.name}")
        scenario_vars.append((scenario, p, r))

    for i in range(n):
        prev = 1.0 if data.initial_on[i] else 0.0
        for t in range(horizon):
            model.addConstr(u[i, t] - prev == v[i, t] - w[i, t])
            model.addConstr(v[i, t] + w[i, t] <= 1)
            prev = u[i, t]
        model.addConstr(w[i, horizon - 1] == 0)
        for t in range(horizon):
            for k in range(t, min(horizon, t + int(data.min_up[i]))):
                model.addConstr(u[i, k] >= v[i, t])
            for k in range(t, min(horizon, t + int(data.min_down[i]))):
                model.addConstr(u[i, k] <= 1 - w[i, t])
        if data.initial_on[i]:
            remaining = max(0, int(data.min_up[i]) - int(data.initial_duration[i]))
            for t in range(remaining):
                model.addConstr(u[i, t] == 1)
        else:
            remaining = max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
            for t in range(remaining):
                model.addConstr(u[i, t] == 0)

    for scenario, p, r in scenario_vars:
        for i in range(n):
            for t in range(horizon):
                model.addConstr(p[i, t] >= data.pmin[i] * u[i, t])
                startup_capacity = data.pmax[i] * u[i, t] - (data.pmax[i] - data.pstart[i]) * v[i, t]
                shutdown_capacity = data.pmax[i] * u[i, t]
                if t + 1 < horizon:
                    shutdown_capacity -= (data.pmax[i] - data.pshut[i]) * w[i, t + 1]
                model.addConstr(p[i, t] + r[i, t] <= startup_capacity)
                model.addConstr(p[i, t] + r[i, t] <= shutdown_capacity)
                if t == 0:
                    model.addConstr(p[i, t] - data.initial_output[i] <= data.ramp_up[i])
                    model.addConstr(data.initial_output[i] - p[i, t] <= data.ramp_down[i])
                else:
                    model.addConstr(p[i, t] - p[i, t - 1] <= data.ramp_up[i])
                    model.addConstr(p[i, t - 1] - p[i, t] <= data.ramp_down[i])
        for t in range(horizon):
            model.addConstr(sum(p[i, t] for i in range(n)) == scenario.load[t])
            model.addConstr(sum(r[i, t] for i in range(n)) >= data.reserve_fraction * scenario.load[t])

    fixed = sum(
        data.cost_c[i] * u[i, t]
        + data.startup_cost[i] * v[i, t]
        + data.shutdown_cost[i] * w[i, t]
        for i in range(n) for t in range(horizon)
    )
    expected_fuel = sum(
        scenario.probability * sum(
            data.cost_a[i] * p[i, t] * p[i, t] + data.cost_b[i] * p[i, t]
            for i in range(n) for t in range(horizon)
        )
        for scenario, p, _r in scenario_vars
    )
    model.setObjective(fixed + expected_fuel, gp.GRB.MINIMIZE)
    model.optimize()
    runtime = time.perf_counter() - started

    status = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
    }.get(model.Status, str(model.Status))
    # ObjVal is a feasible objective, even when OPTIMAL means a nonzero MIP gap.
    # Only ObjBound supplies the independent reference lower bound.
    lower_bound = float("-inf")
    if status not in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}:
        lower_bound = float(model.ObjBound)
        if abs(lower_bound) >= gp.GRB.INFINITY:
            lower_bound = float("-inf")

    if model.SolCount == 0:
        return FullResult(status, float("inf"), float("inf"), float("inf"), np.empty((n, horizon)), np.empty((n, horizon)), np.empty((n, horizon)), [], float("inf"), runtime, model, lower_bound)

    u_value = np.rint(np.asarray(u.X, dtype=float)).astype(int)
    v_value = np.rint(np.asarray(v.X, dtype=float)).astype(int)
    w_value = np.rint(np.asarray(w.X, dtype=float)).astype(int)
    dispatches = []
    expected_cost = 0.0
    for scenario, p, r in scenario_vars:
        p_value = np.asarray(p.X, dtype=float)
        r_value = np.asarray(r.X, dtype=float)
        fuel = float(np.sum(data.cost_a[:, None] * p_value**2 + data.cost_b[:, None] * p_value))
        expected_cost += scenario.probability * fuel
        dispatches.append(ScenarioDispatch(scenario.name, scenario.probability, p_value, r_value, fuel))
    fixed_cost = float(
        np.sum(data.cost_c[:, None] * u_value)
        + np.sum(data.startup_cost[:, None] * v_value)
        + np.sum(data.shutdown_cost[:, None] * w_value)
    )
    return FullResult(
        status=status,
        objective=fixed_cost + expected_cost,
        fixed_cost=fixed_cost,
        expected_fuel_cost=expected_cost,
        u=u_value,
        v=v_value,
        w=w_value,
        scenarios=dispatches,
        mip_gap=float(model.MIPGap) if model.IsMIP else 0.0,
        runtime=runtime,
        model=model,
        lower_bound=lower_bound,
    )
