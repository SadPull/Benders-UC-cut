"""Independent physical and cost validation for complete SUC schedules."""
from types import SimpleNamespace

import numpy as np

from .common import commitment_is_feasible


def schedule_object(row):
    return SimpleNamespace(objective=row["objective"],
        **{k: np.asarray(row[k]) for k in ("u", "v", "w")},
        scenarios=[SimpleNamespace(name=s["name"], objective=s.get("objective", s.get("fuel_cost")),
            p=np.asarray(s["p"]), r=np.asarray(s["r"])) for s in row["scenarios"]])


def validate_schedule(data, scenarios, result):
    """Validate a complete physical schedule independently of experiment budgets."""
    if not np.isfinite(result.objective):
        return {"feasible_solution": False, "trace_valid": True}
    if not commitment_is_feasible(data, scenarios, result.u, result.v, result.w):
        raise AssertionError("returned commitment is infeasible")
    residuals = {}
    fuel = 0.0
    next_shutdown = np.column_stack((result.w[:, 1:], np.zeros(data.n_units)))
    startup_cap = data.pmax[:, None] * result.u - (data.pmax - data.pstart)[:, None] * result.v
    shutdown_cap = data.pmax[:, None] * result.u - (data.pmax - data.pshut)[:, None] * next_shutdown
    if len(result.scenarios) != len(scenarios):
        raise AssertionError("missing scenario dispatch")
    for scenario, solution in zip(scenarios, result.scenarios):
        p, r = solution.p, solution.r
        if solution.name != scenario.name or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise AssertionError("invalid scenario dispatch")
        previous = np.column_stack((data.initial_output, p[:, :-1]))
        checks = {
            "balance": float(np.max(np.abs(p.sum(axis=0) - scenario.load))),
            "reserve": float(np.max(data.reserve_fraction * scenario.load - r.sum(axis=0))),
            "pmin": float(np.max(data.pmin[:, None] * result.u - p)),
            "startup_capacity": float(np.max(p + r - startup_cap)),
            "shutdown_capacity": float(np.max(p + r - shutdown_cap)),
            "ramp_up": float(np.max(p - previous - data.ramp_up[:, None])),
            "ramp_down": float(np.max(previous - p - data.ramp_down[:, None])),
            "nonnegative": float(max(np.max(-p), np.max(-r))),
        }
        residuals[scenario.name] = {key: max(0.0, value) for key, value in checks.items()}
        scenario_fuel = float(np.sum(data.cost_a[:, None] * p**2 + data.cost_b[:, None] * p))
        if abs(scenario_fuel - solution.objective) > 1e-4:
            raise AssertionError("scenario cost mismatch")
        fuel += scenario.probability * scenario_fuel
    maximum = max(value for row in residuals.values() for value in row.values())
    fixed = float(np.sum(data.cost_c[:, None] * result.u) + np.sum(data.startup_cost[:, None] * result.v)
                  + np.sum(data.shutdown_cost[:, None] * result.w))
    if maximum > 1e-5 or abs(fixed + fuel - result.objective) > 1e-4:
        raise AssertionError("dispatch or total cost validation failed")
    return {"feasible_solution": True, "trace_valid": True, "max_residual_mw": maximum,
            "scenario_residuals": residuals, "cost_residual": abs(fixed + fuel - result.objective)}
