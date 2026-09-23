from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .common import Scenario, require_gurobi
from .data import UCData
from .master import BendersCut


@dataclass
class ExactMasterConfig:
    """Settings for the exact Gurobi Benders master problem."""

    time_limit: float = 60.0
    mip_gap: float = 1e-6
    output_flag: int = 0


@dataclass
class ExactMasterResult:
    status: str
    objective: float
    bound: float
    mip_gap: float
    runtime: float
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    w: np.ndarray | None = None
    theta: float | None = None
    model: Any = None


def _status_name(gp, status: int) -> str:
    return {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status, str(status))


def _cut_terms(gp, cut: BendersCut, u, v, w):
    """Linear x-term of a Benders cut, skipping exact zeros."""

    terms = []
    for i in range(cut.beta_u.shape[0]):
        for t in range(cut.beta_u.shape[1]):
            if cut.beta_u[i, t]:
                terms.append(float(cut.beta_u[i, t]) * u[i, t])
            if cut.beta_v[i, t]:
                terms.append(float(cut.beta_v[i, t]) * v[i, t])
            if cut.beta_w[i, t]:
                terms.append(float(cut.beta_w[i, t]) * w[i, t])
    return gp.quicksum(terms)


def solve_exact_master(
    data: UCData,
    scenarios: Sequence[Scenario],
    cuts: Iterable[BendersCut],
    config: ExactMasterConfig | None = None,
) -> ExactMasterResult:
    """Solve the Benders master problem exactly with Gurobi.

    The model carries the same first-stage constraints as the full two-stage
    model plus the accumulated Benders cuts, with a continuous theta.  Its
    objective value is therefore a valid lower bound for the original
    stochastic unit commitment problem whenever the reported bound was proven
    by the solver.
    """

    config = config or ExactMasterConfig()
    started = time.perf_counter()
    gp = require_gurobi()
    n, horizon = data.n_units, data.horizon
    cut_list = list(cuts)

    model = gp.Model("suc_benders_exact_master")
    model.Params.OutputFlag = config.output_flag
    model.Params.MIPGap = config.mip_gap
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit

    u = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="u")
    v = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="v")
    w = model.addMVar((n, horizon), vtype=gp.GRB.BINARY, name="w")
    theta = model.addVar(lb=0.0, name="theta")

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

    for scenario in scenarios:
        for t in range(horizon):
            capacity = []
            for i in range(n):
                expr = data.pmax[i] * u[i, t] - (data.pmax[i] - data.pstart[i]) * v[i, t]
                if t + 1 < horizon:
                    expr = expr - (data.pmax[i] - data.pshut[i]) * w[i, t + 1]
                capacity.append(expr)
            model.addConstr(
                gp.quicksum(capacity) >= (1.0 + data.reserve_fraction) * float(scenario.load[t])
            )

    for cut in cut_list:
        terms = _cut_terms(gp, cut, u, v, w)
        if cut.kind == "optimality":
            model.addConstr(theta >= float(cut.alpha) + terms)
        else:
            model.addConstr(float(cut.alpha) + terms <= 0)

    fixed = gp.quicksum(
        float(data.cost_c[i]) * u[i, t]
        + float(data.startup_cost[i]) * v[i, t]
        + float(data.shutdown_cost[i]) * w[i, t]
        for i in range(n)
        for t in range(horizon)
    )
    model.setObjective(fixed + theta, gp.GRB.MINIMIZE)
    model.optimize()
    runtime = time.perf_counter() - started

    status = _status_name(gp, model.Status)
    objective = float(model.ObjVal) if model.SolCount > 0 else float("inf")
    if model.Status == gp.GRB.OPTIMAL:
        bound = objective
    elif model.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
        # An empty master relaxation means the original problem is infeasible.
        bound = float("inf")
    else:
        # Only a proven bound may be reported; a truncated solve returns the
        # best proven bound, never the incumbent objective.
        bound = float(model.ObjBound)
        tolerance = 1e-6 * max(1.0, abs(objective)) if math.isfinite(objective) else 0.0
        if not math.isfinite(bound) or (math.isfinite(objective) and bound > objective + tolerance):
            bound = float("-inf")

    mip_gap = float(model.MIPGap) if model.IsMIP else 0.0
    if math.isinf(bound) and bound > 0:
        mip_gap = float("inf")

    if model.SolCount == 0:
        return ExactMasterResult(
            status=status,
            objective=objective,
            bound=bound,
            mip_gap=mip_gap,
            runtime=runtime,
            model=model,
        )

    return ExactMasterResult(
        status=status,
        objective=objective,
        bound=bound,
        mip_gap=mip_gap,
        runtime=runtime,
        u=np.rint(np.asarray(u.X, dtype=float)).astype(int),
        v=np.rint(np.asarray(v.X, dtype=float)).astype(int),
        w=np.rint(np.asarray(w.X, dtype=float)).astype(int),
        theta=float(theta.X),
        model=model,
    )
