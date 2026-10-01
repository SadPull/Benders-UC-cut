"""Independent, bounded original-block MIQP control; never a research backend."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .attribution import validate_result, write_json
from .candidates import repair_commitment
from .common import require_gurobi
from .paper_model import Block, IndexedUC, binary, key
from .paper_solver import coordinate
from .solver import _construct_feasible_candidate, _fixed_cost
from .subproblem import solve_scenario


def pack(point):
    return [[list(k), float(v)] for k, v in point.items()]


def unpack(point):
    return {tuple(k): v for k, v in point}


def finite(value):
    return float(value) if np.isfinite(value) and abs(value) < 1e100 else None


def solve_block(block, costs, deadline, limit=30.):
    """Solve the original local MIQP; bounds and incumbents remain distinct."""
    started = time.perf_counter()
    empty = dict(status="BUDGET_EXHAUSTED", objective=None, bound=None, gap=None,
                 point=None, certified=False, residual=None, seconds=0.)
    if started >= deadline:
        return empty
    gp = require_gurobi()
    m = gp.Model("exact_block_control")
    try:
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.Seed = 7
        m.Params.MIPGap = 1e-8
        m.Params.MIPGapAbs = 1e-4
        x = {k: m.addVar(lb=block.uc.bounds[k][0], ub=block.uc.bounds[k][1],
                        vtype=gp.GRB.BINARY if binary(k) else gp.GRB.CONTINUOUS)
             for k in block.keys}
        for row in block.rows:
            expr = row.c + gp.quicksum(v*x[k] for k, v in row.a.items())
            m.addConstr(expr == 0 if row.equality else expr <= 0)
        m.setObjective(gp.quicksum(costs[k]*x[k] + block.uc.quadratic.get(k, 0.)*x[k]*x[k]
                                  for k in block.keys), gp.GRB.MINIMIZE)
        remaining = min(limit-(time.perf_counter()-started), deadline-time.perf_counter())
        if remaining <= 0:
            return {**empty, "seconds": time.perf_counter()-started}
        m.Params.TimeLimit = remaining
        m.optimize()
        statuses = {gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.TIME_LIMIT: "TIME_LIMIT",
                    gp.GRB.INFEASIBLE: "INFEASIBLE", gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    gp.GRB.INTERRUPTED: "INTERRUPTED"}
        result = {**empty, "status": statuses.get(m.Status, str(m.Status)),
                  "bound": finite(m.ObjBound), "seconds": time.perf_counter()-started}
        if m.SolCount:
            point = {k: float(v.X) for k, v in x.items()}
            residual = max([0.] + [abs(r.value(point)) if r.equality else r.value(point)
                                   for r in block.rows] +
                           [abs(point[k]-round(point[k])) for k in block.bkeys] +
                           [max(block.uc.bounds[k][0]-v, v-block.uc.bounds[k][1]) for k, v in point.items()])
            objective = sum(costs[k]*v + block.uc.quadratic.get(k, 0.)*v*v for k, v in point.items())
            if residual > 1e-5 or not np.isclose(objective, m.ObjVal, atol=1e-4, rtol=1e-9):
                raise AssertionError("invalid exact-block incumbent")
            result.update(objective=float(m.ObjVal), point=pack(point), gap=finite(m.MIPGap),
                          residual=residual, certified=m.Status == gp.GRB.OPTIMAL)
        return result
    finally:
        m.dispose()


def aggregate(records, count, constant):
    complete = len(records) == count
    lower = (sum(r["bound"] for r in records)+constant
             if complete and all(r["bound"] is not None for r in records) else None)
    upper = (sum(r["objective"] for r in records)+constant
             if complete and all(r["objective"] is not None for r in records) else None)
    return dict(lower=lower, upper=upper,
                certified=complete and all(r["certified"] for r in records))


class Budget:
    """Persist elapsed budget. Interrupted active time (including downtime) is charged."""
    def __init__(self, path, state, total):
        self.path, self.state, self.total = path, state, total
        ledger = state.setdefault("budget", {"used": 0., "active_since": None})
        if ledger["active_since"] is not None:
            ledger["used"] = min(total, ledger["used"] + max(0., time.time()-ledger["active_since"]))
        self.started = time.perf_counter()
        self.base = ledger["used"]
        self.deadline = self.started + max(0., total-self.base)
        self.save()

    def elapsed(self):
        return self.base + time.perf_counter()-self.started

    def available(self):
        return time.perf_counter() < self.deadline

    def save(self, finished=False):
        self.state["budget"] = {"used": self.elapsed(), "active_since": None if finished else time.time()}
        write_json(self.path, self.state)


def evaluate(data, scenarios, candidate, deadline):
    """Validate one complete candidate with the existing physical/cost validator."""
    if candidate is None or time.perf_counter() >= deadline:
        return None
    solutions = []
    for scenario in scenarios:
        if time.perf_counter() >= deadline:
            return None
        r = solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w, deadline=deadline, threads=1)
        if not r.feasible or "TIME_LIMIT" in r.status:
            return None
        solutions.append(SimpleNamespace(name=scenario.name, objective=r.objective, p=r.p, r=r.r))
    objective = _fixed_cost(data, candidate.u, candidate.v, candidate.w) + sum(
        s.probability*r.objective for s, r in zip(scenarios, solutions))
    event = dict(cache_hit=False, evaluation_count=1, commitment_id="validation", incumbent=objective)
    result = SimpleNamespace(objective=objective, u=candidate.u, v=candidate.v, w=candidate.w,
                             scenarios=solutions, diagnostics={"evaluations": [event], "evaluation_count": 1})
    validation = validate_result(data, scenarios, result)
    return dict(objective=objective, u=candidate.u.tolist(), v=candidate.v.tolist(), w=candidate.w.tolist(),
                scenarios=[dict(name=r.name, objective=r.objective, p=r.p.tolist(), r=r.r.tolist()) for r in solutions],
                validation=validation)


def closed_loop(data, scenarios, hours, state, budget, block_limit=30., max_rounds=10):
    uc = IndexedUC(data, scenarios, hours)
    blocks = [Block(uc, i) for i in range(len(uc.blocks))]
    state.setdefault("prices", [0.]*len(uc.coupling))
    state.setdefault("planes", [])
    state.setdefault("rounds", [])
    state.setdefault("evaluations", [])
    state.setdefault("best_lower", None)
    if "initialized" not in state:
        seed = _construct_feasible_candidate(data, scenarios) if budget.available() else None
        value = evaluate(data, scenarios, seed, budget.deadline)
        if value:
            state["evaluations"].append({**value, "elapsed": budget.elapsed(), "origin": "initial"})
        state["initialized"] = True
        budget.save()
    for outer in range(len(state["rounds"])+1, max_rounds+1):
        if not budget.available():
            break
        prices = np.asarray(state["prices"])
        costs, constant = uc.coefficients(prices)
        pending = state.setdefault("pending", {"outer": outer, "blocks": []})
        for block in blocks[len(pending["blocks"]):]:
            if not budget.available():
                break
            pending["blocks"].append(solve_block(block, costs, budget.deadline, block_limit))
            budget.save()
        bounds = aggregate(pending["blocks"], len(blocks), constant)
        row = dict(outer=outer, prices=prices.tolist(), blocks=pending["blocks"], **bounds)
        if bounds["lower"] is not None:
            state["best_lower"] = max(state["best_lower"] if state["best_lower"] is not None else -np.inf, bounds["lower"])
        row["best_lower"] = state["best_lower"]
        row["bound_elapsed"] = budget.elapsed()
        row["boundary_fraction"] = float(np.mean(np.abs(prices) >= 10-1e-7)) if len(prices) else 0.
        row["coordination_upper"] = None
        if bounds["upper"] is not None:
            point = {k: v for r in pending["blocks"] for k, v in unpack(r["point"]).items()}
            original, residual = uc.original_cost(point), uc.residuals(point)
            if not np.isclose(bounds["upper"], original+uc.cost_scale*np.dot(prices, residual), atol=1e-4, rtol=1e-9):
                raise AssertionError("Lagrangian allocation mismatch")
            # Replacing a pending round after interruption must not duplicate its plane.
            state["planes"] = state["planes"][:outer-1] + [[original, residual.tolist()]]
            row["plane_kind"] = "certified_within_tolerance" if bounds["certified"] else "approximate"
            if budget.available():
                u = np.array([[round(point[key("u", i, t)]) for t in range(data.horizon)] for i in range(data.n_units)])
                candidate = repair_commitment(data, scenarios, u)
                value = evaluate(data, scenarios, candidate, budget.deadline)
                if value:
                    state["evaluations"].append({**value, "elapsed": budget.elapsed(), "origin": f"outer_{outer}"})
            if budget.available():
                update = coordinate(uc, state["planes"], budget.deadline)
                if update is not None:
                    state["prices"] = update[0].tolist()
                    row["coordination_upper"] = update[1]
        row["elapsed"] = budget.elapsed()
        state["rounds"].append(row)
        del state["pending"]
        budget.save()
        if bounds["upper"] is None or bounds["lower"] is None:
            state["status"] = "INCOMPLETE_ROUND"
            break
        upper, lower = row["coordination_upper"], state["best_lower"]
        if upper is not None:
            tolerance = max(1e-4, 1e-8*max(1., abs(upper), abs(lower)))
            if upper < lower-tolerance:
                raise AssertionError("coordination bound below certified lower bound")
            if upper-lower <= tolerance:
                state["status"] = "BOX_DUAL_CONVERGED"
                break
        if row["coordination_upper"] is None:
            state["status"] = "COORDINATION_UNAVAILABLE"
            break
    state.setdefault("status", "TIME_LIMIT" if not budget.available() else "MAX_ITERATIONS")
    state["finished"] = True
    budget.save(finished=True)
