"""Hourly recourse bounds (fuel plus cold premium); relax inter-hour continuous rows."""
# 模块职责：构造逐小时连续补偿费用的松弛下界，包含概率加权燃料及共享冷启动差额。
# SMS 分支的连续补偿还包括缺电和弃电罚费用，因此小时补偿不能一律当作纯燃料。
# 只放松跨小时连续耦合；与当前小时相关的历史二进制状态仍保留在该小时割的支持中。
import hashlib
from types import SimpleNamespace as NS

import numpy as np

from .benders import dispatch, recourse_model


# 保存各小时子问题、局部二进制支持、已评估锚点和认证割池。
class HourlyBounds:
    # 保留连续变量只属于单个小时的行，移除跨小时爬坡等连续耦合，并抽取相关二进制列。
    # 纯离散约束由全天方案检查处理；本子问题只计算连续补偿费用。
    def __init__(self, full):
        self.full, self.parts, self.cuts, self.identities = full, [], [], set()
        X, Y = full.binary, full.continuous
        counts = full.A[:, Y].getnnz(axis=1)
        for hour in full.times:
            variables = Y[full.time[Y] == hour]
            local_counts = full.A[:, variables].getnnz(axis=1)
            # 局部与全天连续非零数相同，才表示该行没有耦合其他小时的连续变量。
            rows = np.flatnonzero((local_counts > 0) & (local_counts == counts))
            indices = np.flatnonzero(full.A[rows][:, X].getnnz(axis=0))
            b = recourse_model(full, rows, X[indices], variables)
            b.constant = 0.
            b.audit_cuts = (bool(getattr(full, 'audit_cuts', False)) or
                           getattr(getattr(full, "d", None), "model_profile", "") == "azema_sms")
            self.parts.append(NS(hour=int(hour), b=b, indices=indices, variables=variables, cache=set()))

    # 从全天调度点提取每小时真实连续费用，返回长度为小时数的向量。
    # 其中冷启动差额属于补偿费用，不能再当作纯燃料与实际启动费重复相加。
    def recourse(self, point):
        return np.array([self.full.obj[p.variables] @ point[p.variables]
                         + self.full.q[p.variables] @ point[p.variables]**2 for p in self.parts])

    # 对给定全天二进制 x，逐小时取基础费用下界与已有割的最大值，再求和返回下界。
    def lower(self, x):
        values = [float(np.minimum(p.b.obj[p.b.continuous]*p.b.lb[p.b.continuous],
                                   p.b.obj[p.b.continuous]*p.b.ub[p.b.continuous]).sum()) for p in self.parts]
        for cut in self.cuts:
            hour = cut["hour"]-1
            values[hour] = max(values[hour], cut["alpha"] + np.asarray(cut["beta"]) @ x[cut["binary_indices"]])
        return float(sum(values))

    # 对未见过的小时二进制支持求解认证松弛，检查锚点下界不超过真实费用，再逐条 yield 新割。
    # remaining 是实时剩余秒数回调；诊断 hour 使用从 1 开始的编号。
    def observe(self, x, source, actual, remaining):
        for part, value in zip(self.parts, actual):
            local = x[part.indices]
            key = local.astype(np.uint8).tobytes()
            if key in part.cache:
                continue
            if remaining() <= 0:
                raise TimeoutError
            r = dispatch(part.b, local, remaining(), certified=True)
            if remaining() <= 0 or r.status == "TIME_LIMIT":
                raise TimeoutError
            if not r.feasible or r.cut is None or not r.cut.certified:
                raise RuntimeError("Hourly relaxation of a validated dispatch failed")
            cut = r.cut
            if cut.alpha + cut.beta @ local > value + 1e-5:
                raise RuntimeError("Hourly lower bound exceeds validated hourly recourse cost")
            part.cache.add(key)
            identity = hashlib.sha256(np.r_[part.hour, cut.alpha, cut.beta].astype("<f8").tobytes()).hexdigest()
            if identity in self.identities:
                continue
            self.identities.add(identity)
            row = dict(cut_id=identity, source=source, hour=part.hour+1,
                       binary_indices=part.indices.tolist(), alpha=float(cut.alpha), beta=cut.beta.tolist())
            if hasattr(cut, "proof"):
                row["proof"] = cut.proof
            self.cuts.append(row)
            yield row
