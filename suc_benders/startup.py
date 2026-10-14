"""Evaluate startup fees from consecutive offline hours, independently of model rows."""
# 模块职责：独立于稀疏模型，按连续停机历史计算实际启动费用与全天固定费用。
# 矩阵为机组 × 小时；冷启动判定含初始停机时长，跨块和跨场景不会重置或重复收费。
import numpy as np


# 从 u 和初始开机状态重建启动时刻，返回每台机组每小时的实际热/冷启动费用。
# 只有启动前连续停机小时数严格大于 min_down+cold_start_hours 才使用冷启动费。
def startup_costs(d, u):
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    starts = np.maximum(u-previous, 0)
    costs = d.startup_cost[:, None]*starts
    if not hasattr(d, "cold_startup_cost"):
        return costs
    offline = np.where(d.initial_on, 0, d.initial_duration).copy()
    threshold = d.min_down+d.cold_start_hours
    for t in range(d.horizon):
        # 先按启动前的停机时长定价，再用本小时开停机状态更新计数，避免差一小时。
        cold = (starts[:, t] > 0) & (offline > threshold)
        costs[cold, t] = d.cold_startup_cost[cold]
        offline = np.where(u[:, t] > 0, 0, offline+1)
    return costs


# 返回全天空载运行费用 + 实际启动费用 + 停机费用，不含任何场景燃料或不平衡罚费用。
def fixed_cost(d, u, w):
    return float(np.sum(d.cost_c[:, None]*u + startup_costs(d, u) + d.shutdown_cost[:, None]*w))
