"""Gurobi adapter. Return numbers and arrays, never live solver objects."""
from types import SimpleNamespace as NS
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

import gurobipy as gp
import numpy as np
from scipy import sparse


_continuous_guard = ContextVar('continuous_guard', default=None)


@contextmanager
def continuous_only(record):
    """An audited scope in which Gurobi must never optimize integer variables."""
    token = _continuous_guard.set(record)
    try:
        yield
    finally:
        _continuous_guard.reset(token)


def optimize(m, seconds, integral=False, *, method=None, log_file=None, farkas=False):
    record = _continuous_guard.get()
    if record is not None and integral:
        raise ValueError('Integer Gurobi optimization is forbidden in paper SA')
    if seconds <= 0:
        raise TimeoutError("No time remaining for optimization")
    if farkas and (integral or np.any(m.q)):
        raise ValueError('Farkas extraction requires a continuous LP')
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as g:
            for name, value in {"Threads": 1, "TimeLimit": seconds, "MIPGap": 1e-6,
                                "FeasibilityTol": 1e-8, "OptimalityTol": 1e-8,
                                "BarConvTol": 1e-10, "Seed": 7}.items():
                g.setParam(name, value)
            if method is not None:
                g.setParam('Method', method)
            if farkas:
                g.setParam('InfUnbdInfo', 1)
                g.setParam('DualReductions', 0)
            if log_file is not None:
                g.setParam('LogToConsole', 0)
                g.setParam('LogFile', str(log_file))
                g.setParam('OutputFlag', 1)
            types = np.full(len(m.obj), gp.GRB.CONTINUOUS)
            if integral:
                types[m.binary] = gp.GRB.BINARY
            x = g.addMVar(len(m.obj), lb=m.lb, ub=m.ub, vtype=types)
            g.setObjective(m.obj @ x + x @ sparse.diags(m.q) @ x)
            constraints = g.addMConstr(m.A, x, np.where(m.eq, "=", "<"), m.rhs)
            g.update()
            call = dict(purpose=getattr(m, 'purpose', 'continuous_relaxation'),
                        integer_variables=int(g.NumIntVars), variables=int(g.NumVars),
                        threads=int(g.Params.Threads), time_limit=float(g.Params.TimeLimit))
            if record is not None:
                if g.NumIntVars:
                    raise ValueError('Integer variables reached the continuous-only Gurobi guard')
                record(dict(call, event='start'))
            started = perf_counter()
            g.optimize()
            if record is not None:
                record(dict(call, event='complete', status=int(g.Status), seconds=perf_counter()-started))
            statuses = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
                        9: "TIME_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL"}
            result = NS(status=statuses.get(g.Status, f"STATUS_{g.Status}"), lower=-np.inf,
                        x=None, objval=np.inf, prices=None, farkas=None)
            if farkas and g.Status == gp.GRB.INFEASIBLE:
                try:
                    result.farkas = np.asarray(constraints.FarkasDual)
                except (AttributeError, gp.GurobiError):
                    pass
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
