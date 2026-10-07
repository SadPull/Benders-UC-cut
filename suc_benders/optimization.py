"""Gurobi adapter. Return numbers and arrays, never live solver objects."""
from types import SimpleNamespace as NS

import gurobipy as gp
import numpy as np
from scipy import sparse


def optimize(m, seconds, integral=False):
    if seconds <= 0:
        raise TimeoutError("No time remaining for optimization")
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as g:
            for name, value in {"Threads": 1, "TimeLimit": seconds, "MIPGap": 1e-6,
                                "FeasibilityTol": 1e-8, "OptimalityTol": 1e-8,
                                "BarConvTol": 1e-10, "Seed": 7}.items():
                g.setParam(name, value)
            types = np.full(len(m.obj), gp.GRB.CONTINUOUS)
            if integral:
                types[m.binary] = gp.GRB.BINARY
            x = g.addMVar(len(m.obj), lb=m.lb, ub=m.ub, vtype=types)
            g.setObjective(m.obj @ x + x @ sparse.diags(m.q) @ x)
            constraints = g.addMConstr(m.A, x, np.where(m.eq, "=", "<"), m.rhs)
            g.optimize()
            statuses = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
                        9: "TIME_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL"}
            result = NS(status=statuses.get(g.Status, f"STATUS_{g.Status}"), lower=-np.inf,
                        x=None, objval=np.inf, prices=None)
            if g.SolCount:
                result.x, result.objval = np.asarray(x.X), float(g.ObjVal)
            if integral:
                try:
                    result.lower = float(g.ObjBound)
                except AttributeError:
                    pass
            elif g.Status == gp.GRB.OPTIMAL:
                prices = -np.asarray(constraints.Pi)
                prices[~m.eq] = np.maximum(0, prices[~m.eq])
                result.prices = prices
                c = m.obj + m.A.T @ prices
                z = np.where(c < 0, m.ub, m.lb).copy()
                curved = m.q > 0
                z[curved] = np.clip(-c[curved] / (2 * m.q[curved]), m.lb[curved], m.ub[curved])
                if np.isfinite(z).all():
                    result.lower = float(-m.rhs @ prices + np.sum(c * z + m.q * z ** 2) - 1e-7)
            return result
