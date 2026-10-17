"""Gurobi adapter. Return numbers and arrays, never live solver objects."""
# 模块职责：将稀疏线性/凸二次模型交给单线程 Gurobi，返回数值结果。
# 连续模式默认放松 binary 列；只有 integral=True 才设为二进制。
from types import SimpleNamespace as NS

import gurobipy as gp
import numpy as np
from scipy import sparse


# 输入稀疏模型与秒预算，返回状态、原始点/目标、对偶价格、下界和可选 Farkas 证据。
# 整数模式取 ObjBound；连续最优时按有界拉格朗日重新计算下界，不把候选 ObjVal 当作整数证明。
def optimize(m, seconds, integral=False, *, method=None, log_file=None, farkas=False,
             warm_start=None, mip_focus=None, target_lower=None, callback=None, model_file=None,
             solution_limit=None):
    if seconds <= 0:
        raise TimeoutError("No time remaining for optimization")
    if farkas and (integral or np.any(m.q)):
        raise ValueError('Farkas extraction requires a continuous LP')
    if not integral and any(value is not None for value in (warm_start, mip_focus, target_lower, solution_limit)):
        raise ValueError('MIP guidance requires integer optimization')
    if solution_limit is not None and (isinstance(solution_limit, bool)
            or not isinstance(solution_limit, (int, np.integer)) or solution_limit <= 0):
        raise ValueError('solution_limit must be a positive integer')
    if warm_start is not None:
        warm_start = np.asarray(warm_start, float)
        if warm_start.shape != m.obj.shape or not np.isfinite(warm_start).all():
            raise ValueError('Invalid MIP start')
        residual = m.A @ warm_start-m.rhs
        residual[m.eq] = abs(residual[m.eq])
        if (max(residual.max(initial=0.), (m.lb-warm_start).max(), (warm_start-m.ub).max()) > 1e-5
                or np.max(abs(warm_start[m.binary]-np.rint(warm_start[m.binary])), initial=0.) > 1e-9):
            raise ValueError('MIP start violates the model')
    if target_lower is not None and not np.isfinite(target_lower):
        raise ValueError('Invalid target bound')
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
            if mip_focus is not None:
                g.setParam('MIPFocus', mip_focus)
                g.setParam('IntFeasTol', 1e-9)
            if target_lower is not None:
                g.setParam('BestBdStop', target_lower-getattr(m, 'constant', 0.)+1e-5)
            if solution_limit is not None:
                g.setParam('SolutionLimit', int(solution_limit))
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
            if warm_start is not None:
                x.Start = warm_start
            g.setObjective(m.obj @ x + x @ sparse.diags(m.q) @ x)
            constraints = g.addMConstr(m.A, x, np.where(m.eq, "=", "<"), m.rhs)
            g.update()
            if model_file is not None:
                g.write(str(model_file))
            if callback is None:
                g.optimize()
            else:
                g.optimize(callback)
            statuses = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
                        9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL", 15: "USER_OBJ_LIMIT"}
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
                # Gurobi 的 <= 行 Pi 符号与这里的拉格朗日乘子相反；等式价格可正可负。
                prices = -np.asarray(constraints.Pi)
                prices[~m.eq] = np.maximum(0, prices[~m.eq])
                result.prices = prices
                c = m.obj + m.A.T @ prices
                # 在边界内逐变量解析最小化 c*z+q*z^2；只有得到有限最小点才返回对偶下界。
                z = np.where(c < 0, m.ub, m.lb).copy()
                curved = m.q > 0
                z[curved] = np.clip(-c[curved] / (2 * m.q[curved]), m.lb[curved], m.ub[curved])
                if np.isfinite(z).all():
                    result.lower = float(-m.rhs @ prices + np.sum(c * z + m.q * z ** 2) - 1e-7)
            return result
