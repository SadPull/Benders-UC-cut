from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .common import require_gurobi, Scenario
from .data import UCData


@dataclass
class ScenarioResult:
    feasible: bool
    status: str
    objective: float = float("inf")
    p: np.ndarray | None = None
    r: np.ndarray | None = None
    cut_alpha: float = 0.0
    cut_beta_u: np.ndarray | None = None
    cut_beta_v: np.ndarray | None = None
    cut_beta_w: np.ndarray | None = None
    balance_dual: np.ndarray | None = None
    reserve_dual: np.ndarray | None = None
    message: str = ""


def _status_name(gp, status: int) -> str:
    return {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    }.get(status, str(status))


def _solve_scenario(
    data: UCData,
    scenario: Scenario,
    u_value: np.ndarray,
    v_value: np.ndarray,
    w_value: np.ndarray,
    output_flag: int = 0,
) -> ScenarioResult:
    """Solve one continuous scenario and return a subgradient cut.

    The commitment-dependent coefficients are attached to every RHS record.
    Gurobi dual multipliers therefore give a subgradient in x=(u,v,w). The
    affine function is anchored at the solved point, which also avoids the
    quadratic-conjugate term needed for an exact QP dual value.
    """

    gp = require_gurobi()
    n, horizon = data.n_units, data.horizon
    model = gp.Model("suc_scenario_subproblem")
    model.Params.OutputFlag = output_flag
    p = model.addMVar((n, horizon), lb=0.0, name="p")
    r = model.addMVar((n, horizon), lb=0.0, name="r")
    records: List[Tuple[object, float, np.ndarray, np.ndarray, np.ndarray]] = []
    balance_constraints: List[object] = []
    reserve_constraints: List[object] = []

    def add(constr, constant, cu=None, cv=None, cw=None):
        records.append(
            (
                constr,
                float(constant),
                np.zeros((n, horizon)) if cu is None else cu,
                np.zeros((n, horizon)) if cv is None else cv,
                np.zeros((n, horizon)) if cw is None else cw,
            )
        )

    for i in range(n):
        for t in range(horizon):
            cu = np.zeros((n, horizon)); cu[i, t] = -data.pmin[i]
            add(model.addConstr(-p[i, t] <= -data.pmin[i] * u_value[i, t]), 0.0, cu=cu)

            cu = np.zeros((n, horizon)); cv = np.zeros((n, horizon)); cw = np.zeros((n, horizon))
            cu[i, t] = data.pmax[i]
            cv[i, t] = -(data.pmax[i] - data.pstart[i])
            if t + 1 < horizon:
                cw[i, t + 1] = -(data.pmax[i] - data.pshut[i])
            rhs = (
                data.pmax[i] * u_value[i, t]
                - (data.pmax[i] - data.pstart[i]) * v_value[i, t]
                - ((data.pmax[i] - data.pshut[i]) * w_value[i, t + 1] if t + 1 < horizon else 0.0)
            )
            add(model.addConstr(p[i, t] + r[i, t] <= rhs), 0.0, cu=cu, cv=cv, cw=cw)

            if t == 0:
                add(model.addConstr(p[i, t] <= data.initial_output[i] + data.ramp_up[i]), data.initial_output[i] + data.ramp_up[i])
                add(model.addConstr(-p[i, t] <= data.ramp_down[i] - data.initial_output[i]), data.ramp_down[i] - data.initial_output[i])
            else:
                add(model.addConstr(p[i, t] - p[i, t - 1] <= data.ramp_up[i]), data.ramp_up[i])
                add(model.addConstr(p[i, t - 1] - p[i, t] <= data.ramp_down[i]), data.ramp_down[i])

    for t in range(horizon):
        balance = model.addConstr(sum(p[i, t] for i in range(n)) == scenario.load[t])
        reserve = model.addConstr(sum(r[i, t] for i in range(n)) >= data.reserve_fraction * scenario.load[t])
        add(balance, float(scenario.load[t]))
        add(reserve, data.reserve_fraction * scenario.load[t])
        balance_constraints.append(balance)
        reserve_constraints.append(reserve)

    fuel = sum(
        data.cost_a[i] * p[i, t] * p[i, t] + data.cost_b[i] * p[i, t]
        for i in range(n) for t in range(horizon)
    )
    model.setObjective(fuel, gp.GRB.MINIMIZE)
    model.optimize()
    status = _status_name(gp, model.Status)
    if model.Status != gp.GRB.OPTIMAL or model.SolCount == 0:
        return _feasibility_cut(data, scenario, u_value, v_value, w_value, output_flag, status)

    beta_u = np.zeros((n, horizon)); beta_v = np.zeros((n, horizon)); beta_w = np.zeros((n, horizon))
    for constr, _constant, cu, cv, cw in records:
        pi = float(constr.Pi)
        beta_u += pi * cu
        beta_v += pi * cv
        beta_w += pi * cw
    objective = float(model.ObjVal)
    alpha = objective - float(np.sum(beta_u * u_value + beta_v * v_value + beta_w * w_value))
    return ScenarioResult(
        feasible=True,
        status=status,
        objective=objective,
        p=np.asarray(p.X, dtype=float),
        r=np.asarray(r.X, dtype=float),
        cut_alpha=alpha,
        cut_beta_u=beta_u,
        cut_beta_v=beta_v,
        cut_beta_w=beta_w,
        balance_dual=np.asarray([float(constr.Pi) for constr in balance_constraints], dtype=float),
        reserve_dual=np.asarray([float(constr.Pi) for constr in reserve_constraints], dtype=float),
    )


def _feasibility_cut(
    data: UCData,
    scenario: Scenario,
    u_value: np.ndarray,
    v_value: np.ndarray,
    w_value: np.ndarray,
    output_flag: int,
    original_status: str,
) -> ScenarioResult:
    """Construct a Benders feasibility cut by minimizing normalized shortfall.

    The auxiliary variables are unnormalized nonnegative shortages s=(load,
    reserve, pmin, capacity, ramp). Normalizing by their RHS coefficients at
    x0 makes phi affine with the same units and keeps multipliers interpretable.
    """

    gp = require_gurobi()
    n, horizon = data.n_units, data.horizon
    model = gp.Model("suc_scenario_feasibility")
    model.Params.OutputFlag = output_flag

    s_balance = model.addMVar(horizon, lb=0.0, name="s_balance")
    s_reserve = model.addMVar(horizon, lb=0.0, name="s_reserve")
    s_pmin = model.addMVar((n, horizon), lb=0.0, name="s_pmin")
    s_capacity = model.addMVar((n, horizon), lb=0.0, name="s_capacity")
    s_ramp = model.addMVar((n, horizon), lb=0.0, name="s_ramp")
    p = model.addMVar((n, horizon), lb=0.0, name="p")
    r = model.addMVar((n, horizon), lb=0.0, name="r")

    records: List[Tuple[object, float, np.ndarray, np.ndarray, np.ndarray]] = []
    zero = np.zeros((n, horizon))

    def add(constr, constant, cu=None, cv=None, cw=None):
        records.append(
            (
                constr,
                float(constant),
                zero if cu is None else cu,
                zero if cv is None else cv,
                zero if cw is None else cw,
            )
        )

    # Keep the same constraint orientation as the optimality subproblem so the
    # recorded x-coefficients and Gurobi dual signs remain directly reusable.
    for i in range(n):
        for t in range(horizon):
            cu = zero.copy()
            cu[i, t] = -data.pmin[i]
            add(
                model.addConstr(-p[i, t] - s_pmin[i, t] <= -data.pmin[i] * u_value[i, t]),
                0.0,
                cu=cu,
            )

            cu = zero.copy()
            cv = zero.copy()
            cw = zero.copy()
            cu[i, t] = data.pmax[i]
            cv[i, t] = -(data.pmax[i] - data.pstart[i])
            if t + 1 < horizon:
                cw[i, t + 1] = -(data.pmax[i] - data.pshut[i])
            rhs = (
                data.pmax[i] * u_value[i, t]
                - (data.pmax[i] - data.pstart[i]) * v_value[i, t]
                - ((data.pmax[i] - data.pshut[i]) * w_value[i, t + 1] if t + 1 < horizon else 0.0)
            )
            add(
                model.addConstr(p[i, t] + r[i, t] - s_capacity[i, t] <= rhs),
                0.0,
                cu=cu,
                cv=cv,
                cw=cw,
            )

            if t == 0:
                add(
                    model.addConstr(p[i, t] - s_ramp[i, t] <= data.initial_output[i] + data.ramp_up[i]),
                    data.initial_output[i] + data.ramp_up[i],
                )
                add(
                    model.addConstr(-p[i, t] - s_ramp[i, t] <= data.ramp_down[i] - data.initial_output[i]),
                    data.ramp_down[i] - data.initial_output[i],
                )
            else:
                add(
                    model.addConstr(p[i, t] - p[i, t - 1] - s_ramp[i, t] <= data.ramp_up[i]),
                    data.ramp_up[i],
                )
                add(
                    model.addConstr(p[i, t - 1] - p[i, t] - s_ramp[i, t] <= data.ramp_down[i]),
                    data.ramp_down[i],
                )

    for t in range(horizon):
        add(
            model.addConstr(sum(p[i, t] for i in range(n)) + s_balance[t] == scenario.load[t]),
            float(scenario.load[t]),
        )
        add(
            model.addConstr(sum(r[i, t] for i in range(n)) + s_reserve[t] >= data.reserve_fraction * scenario.load[t]),
            data.reserve_fraction * scenario.load[t],
        )

    model.setObjective(
        s_balance.sum() + s_reserve.sum() + s_pmin.sum() + s_capacity.sum() + s_ramp.sum(),
        gp.GRB.MINIMIZE,
    )
    model.optimize()
    status = _status_name(gp, model.Status)
    if model.Status != gp.GRB.OPTIMAL or model.SolCount == 0:
        return ScenarioResult(False, f"FEASIBILITY_{status}", message="feasibility LP failed")

    beta_u = np.zeros((n, horizon))
    beta_v = np.zeros((n, horizon))
    beta_w = np.zeros((n, horizon))
    for constr, _constant, cu, cv, cw in records:
        pi = float(constr.Pi)
        beta_u += pi * cu
        beta_v += pi * cv
        beta_w += pi * cw
    phi = float(model.ObjVal)
    alpha = phi - float(np.sum(beta_u * u_value + beta_v * v_value + beta_w * w_value))
    return ScenarioResult(
        feasible=False,
        status=original_status,
        objective=float("inf"),
        cut_alpha=alpha,
        cut_beta_u=beta_u,
        cut_beta_v=beta_v,
        cut_beta_w=beta_w,
        message=f"feasibility shortfall={phi:.6g}",
    )

def solve_scenario(*args, **kwargs) -> ScenarioResult:
    return _solve_scenario(*args, **kwargs)


def cut_value(result: ScenarioResult, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return float(
        result.cut_alpha
        + np.sum(result.cut_beta_u * u)
        + np.sum(result.cut_beta_v * v)
        + np.sum(result.cut_beta_w * w)
    )


def aggregate_optimality_cut(results: list[ScenarioResult], scenarios: list[Scenario]) -> ScenarioResult:
    if not results:
        raise ValueError("at least one scenario result is required")
    alpha = sum(scenario.probability * result.cut_alpha for scenario, result in zip(scenarios, results))
    beta_u = sum(scenario.probability * result.cut_beta_u for scenario, result in zip(scenarios, results))
    beta_v = sum(scenario.probability * result.cut_beta_v for scenario, result in zip(scenarios, results))
    beta_w = sum(scenario.probability * result.cut_beta_w for scenario, result in zip(scenarios, results))
    return ScenarioResult(
        feasible=True,
        status="AGGREGATED",
        objective=sum(scenario.probability * result.objective for scenario, result in zip(scenarios, results)),
        cut_alpha=float(alpha),
        cut_beta_u=np.asarray(beta_u),
        cut_beta_v=np.asarray(beta_v),
        cut_beta_w=np.asarray(beta_w),
    )
