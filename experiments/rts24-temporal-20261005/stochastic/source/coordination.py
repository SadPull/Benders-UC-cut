"""Upper cutting planes for the concave Lagrangian dual (residual <= 0)."""
import math
import time

from .common import require_gurobi


ABSOLUTE_TOLERANCE = 1e-5


def converged(lower, upper, relative, *, reference=None):
    if not all(math.isfinite(v) for v in (lower, upper)):
        return False
    scale = upper if reference is None else reference
    tolerance = ABSOLUTE_TOLERANCE + relative * max(1., abs(scale))
    return -ABSOLUTE_TOLERANCE <= upper-lower <= tolerance


def cutting_plane(block, cross, point):
    """Evaluate a fixed feasible primal point, never the inner lower bound.

    The intercept is original cost; each cross-row constant belongs to its
    earliest block. Thus summing these planes counts the constant exactly once.
    """
    if not block.discrete_checks(point)["feasible"] or any(
        (abs(row.value(point)) if row.equality else row.value(point)) > 1e-6
        for row in block.rows
    ) or any(not math.isfinite(point[k]) or point[k] < block.uc.bounds[k][0]-1e-6
             or point[k] > block.uc.bounds[k][1]+1e-6 for k in block.keys):
        raise ValueError("coordination requires a feasible local dispatch")
    intercept = sum(block.uc.linear[k]*point[k] + block.uc.quadratic.get(k, 0.)*point[k]**2
                    for k in block.keys)
    keys = set(block.keys)
    slope = {}
    for i in cross:
        row = block.uc.rows[i]
        value = sum(v*point[k] for k, v in row.a.items() if k in keys)
        if min(k[2] for k in row.a) in block.times:
            value += row.c
        if value:
            slope[i] = value
    return dict(intercept=intercept, slope=slope)


def coordinate(indexed, cross, planes, incumbent, deadline):
    gp = require_gurobi()
    model = gp.Model("lagrangian_cutting_plane")
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(.001, deadline-time.perf_counter())
        prices = {i: model.addVar(lb=-gp.GRB.INFINITY if indexed.rows[i].equality else 0.) for i in cross}
        theta = [model.addVar(lb=-gp.GRB.INFINITY) for _ in planes]
        for j, pool in enumerate(planes):
            for plane in pool:
                model.addConstr(theta[j] <= plane["intercept"] + gp.quicksum(v*prices[i] for i, v in plane["slope"].items()))
        model.addConstr(gp.quicksum(theta) <= incumbent)
        model.setObjective(gp.quicksum(theta), gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"coordination LP not optimal: {model.Status}")
        multipliers = [0.] * len(indexed.rows)
        for i, variable in prices.items():
            multipliers[i] = float(variable.X) if indexed.rows[i].equality else max(0., float(variable.X))
        return dict(upper=float(model.ObjVal), multipliers=multipliers, seconds=float(model.Runtime))
    finally:
        model.dispose()
