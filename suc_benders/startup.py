"""Evaluate startup fees from consecutive offline hours, independently of model rows."""
import numpy as np


def startup_costs(d, u):
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    starts = np.maximum(u-previous, 0)
    costs = d.startup_cost[:, None]*starts
    if not hasattr(d, "cold_startup_cost"):
        return costs
    offline = np.where(d.initial_on, 0, d.initial_duration).copy()
    threshold = d.min_down+d.cold_start_hours
    for t in range(d.horizon):
        cold = (starts[:, t] > 0) & (offline > threshold)
        costs[cold, t] = d.cold_startup_cost[cold]
        offline = np.where(u[:, t] > 0, 0, offline+1)
    return costs


def fixed_cost(d, u, w):
    return float(np.sum(d.cost_c[:, None]*u + startup_costs(d, u) + d.shutdown_cost[:, None]*w))
