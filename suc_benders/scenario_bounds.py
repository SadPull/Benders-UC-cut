"""Scenario-disaggregated quadratic recourse certificates, with all ramp rows."""
# 模块职责：按场景分解连续补偿费用，复用全天调度的对偶价格生成场景认证割。
# 保留全部跨小时爬坡行；仅支持连续变量可以完全分成各场景 p/r 的固定启动费用模型。
import hashlib
from types import SimpleNamespace as NS

import numpy as np

from .benders import recourse_model


# 维护场景子模型及其有界拉格朗日割池，目标和对偶均沿用联合调度的概率权重。
class ScenarioBounds:
    # 按场景提取 p/r 列及其约束，检查不存在跨场景连续耦合或未归属的共享连续变量。
    # 拒绝含共享冷启动差额的模型，避免将同一连续费用重复分入多个场景。
    def __init__(self, full, model):
        self.full, self.parts, self.cuts, self.identities = full, [], [], set()
        X, Y = full.binary, full.continuous
        groups = [np.r_[model.ix.p[:, :, k].ravel(), model.ix.r[:, :, k].ravel()]
                  for k in range(len(model.scenarios))]
        if not np.array_equal(np.sort(np.concatenate(groups)), Y):
            raise ValueError("Scenario cuts require fixed startup and separate p/r recourse")
        counts = full.A[:, Y].getnnz(axis=1)
        for k, variables in enumerate(groups):
            local_counts = full.A[:, variables].getnnz(axis=1)
            if ((local_counts > 0) & (local_counts != counts)).any():
                raise ValueError("Cross-scenario continuous constraint")
            rows = np.flatnonzero(local_counts)
            b = recourse_model(full, rows, X, variables)
            self.parts.append(NS(scenario=k, b=b, rows=rows, variables=variables))

    # 从全天点读取各场景已按概率加权的连续费用，返回场景向量，不再额外乘概率。
    def recourse(self, point):
        return np.array([self.full.obj[p.variables] @ point[p.variables]
            + self.full.q[p.variables] @ point[p.variables]**2 for p in self.parts])

    # 逐场景取变量边界基础下界与已有割的最大值，再求和得到给定 x 的补偿费用下界。
    def lower(self, x):
        values = [float(np.minimum(p.b.obj[p.b.continuous]*p.b.lb[p.b.continuous],
                                   p.b.obj[p.b.continuous]*p.b.ub[p.b.continuous]).sum()) for p in self.parts]
        for cut in self.cuts:
            k = cut['scenario']
            values[k] = max(values[k], cut['alpha']+np.asarray(cut['beta']) @ x)
        return float(sum(values))

    # 取联合调度的本场景对偶价格，在变量边界内解析最小化拉格朗日函数并保存 alpha/beta 及证明。
    # 锚点既不能超过真实场景费用，也不能过于松弛；非法乘子直接拒绝。
    def observe(self, x, evaluation, source, elapsed):
        actual = self.recourse(evaluation.point)
        for part, value in zip(self.parts, actual):
            b, Y = part.b, part.b.continuous
            prices = np.asarray(evaluation.prices[part.rows]).copy()
            if (prices[~b.eq] < 0).any() or not np.isfinite(prices).all():
                raise ValueError("Invalid scenario multipliers")
            c = b.obj[Y]+np.asarray(b.A[:, Y].T @ prices)
            q, lo, hi = b.q[Y], b.lb[Y], b.ub[Y]
            # 线性项取最优端点；正二次项取区间内的 -c/(2*q)，共同构成有界拉格朗日下界。
            z = np.where(c < 0, hi, lo).copy()
            curved = q > 0
            z[curved] = np.clip(-c[curved]/(2*q[curved]), lo[curved], hi[curved])
            alpha = float(c @ z+q @ z**2-b.rhs @ prices-1e-5)
            beta = np.asarray(b.A[:, b.binary].T @ prices)
            anchor = alpha+beta @ x
            if anchor > value+1e-5 or value-anchor > 1e-3:
                raise RuntimeError("Scenario certificate is invalid or loose at its anchor")
            identity = hashlib.sha256(np.r_[part.scenario, alpha, beta].astype('<f8').tobytes()).hexdigest()
            if identity in self.identities:
                continue
            self.identities.add(identity)
            active = np.flatnonzero(prices)
            self.cuts.append(dict(cut_id=identity, scenario=part.scenario, source=source,
                optimality=True, alpha=alpha, beta=beta.tolist(), anchor_value=float(anchor),
                actual_recourse=float(value), elapsed_seconds=elapsed,
                proof=dict(kind='bounded_lagrangian', indices=active.tolist(), prices=prices[active].tolist())))
