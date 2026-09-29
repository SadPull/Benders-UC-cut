from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .data import UCData


def require_gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "gurobipy is required. Install it with `python -m pip install gurobipy==13.0.1`."
        ) from exc
    return gp


def require_kaiwu():
    try:
        import kaiwu as kw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kaiwu is required for the local simulated-annealing backend."
        ) from exc
    return kw


def get_kaiwu():
    return require_kaiwu()


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    load: np.ndarray


def commitment_is_feasible(
    data: UCData, scenarios: Sequence[Scenario], u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> bool:
    """Check all first-stage constraints exactly, independently of QUBO penalties."""

    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)
    n, horizon = data.n_units, data.horizon
    if u.shape != (n, horizon) or v.shape != u.shape or w.shape != u.shape:
        return False
    if np.any((u != 0) & (u != 1)) or np.any((v != 0) & (v != 1)) or np.any((w != 0) & (w != 1)):
        return False

    for i in range(n):
        prev = int(data.initial_on[i])
        for t in range(horizon):
            if u[i, t] - prev != v[i, t] - w[i, t] or v[i, t] + w[i, t] > 1:
                return False
            prev = int(u[i, t])
        if w[i, horizon - 1] != 0:
            return False
        for t in range(horizon):
            if v[i, t] and np.any(u[i, t : min(horizon, t + int(data.min_up[i]))] != 1):
                return False
            if w[i, t] and np.any(u[i, t : min(horizon, t + int(data.min_down[i]))] != 0):
                return False
        if data.initial_on[i]:
            remaining = max(0, int(data.min_up[i]) - int(data.initial_duration[i]))
        else:
            remaining = max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
        if np.any(u[i, :remaining] != (1 if data.initial_on[i] else 0)):
            return False

    for scenario in scenarios:
        reserve = (1.0 + data.reserve_fraction) * np.asarray(scenario.load)
        for t in range(horizon):
            capacity = sum(
                data.pmax[i] * u[i, t]
                - (data.pmax[i] - data.pstart[i]) * v[i, t]
                - ((data.pmax[i] - data.pshut[i]) * w[i, t + 1] if t + 1 < horizon else 0.0)
                for i in range(n)
            )
            if capacity + 0.5 < reserve[t]:
                return False
    return True
