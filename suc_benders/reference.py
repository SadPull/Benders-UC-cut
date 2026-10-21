"""Solve and independently validate the full-model Gurobi reference."""
from time import perf_counter

import numpy as np

from .model import model
from .optimization import optimize
from .validation import unpack, validate


def solve_reference(data, scenarios, seconds, log_file=None):
    started = perf_counter()
    reference = dict(status="REFERENCE_FAILED", objective=None, validation=None,
                     time_limit_seconds=seconds)
    try:
        full = model(data, scenarios)
        raw = optimize(full, seconds, integral=True, log_file=log_file)
        reference["status"] = raw.status
        # Gurobi 自身的界与间隙只说明参考方案离最优有多远，不参与停止判断。
        if np.isfinite(raw.lower) and abs(raw.lower) < 1e100:
            reference["gurobi_bound"] = float(raw.lower)
        if raw.x is None:
            reference["message"] = "Gurobi 未返回可校验的参考方案"
        else:
            point = raw.x.copy()
            point[full.binary] = np.rint(point[full.binary])
            plan = unpack(full, point)
            if not np.isfinite([plan.objective, raw.objval]).all() or abs(plan.objective-raw.objval) > 1e-4:
                raise ValueError("参考方案重算成本与 Gurobi 目标值不一致")
            reference["validation"] = validate(data, scenarios, plan)
            if not reference["validation"]["feasible"]:
                raise ValueError("参考方案未通过独立校验")
            reference.update(vars(plan))
            if "gurobi_bound" in reference:
                reference["gurobi_gap_percent"] = (100*max(0., plan.objective-reference["gurobi_bound"])
                                                   / max(1., abs(plan.objective)))
    except Exception as error:
        reference.update(status="REFERENCE_FAILED", objective=None,
                         message=f"{type(error).__name__}: {error}")
    reference["elapsed_seconds"] = perf_counter()-started
    return reference
