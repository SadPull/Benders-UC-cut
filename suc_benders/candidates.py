from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .common import Scenario, commitment_capacity, commitment_is_feasible
from .data import UCData
from .master import MasterCandidate


def commitment_key(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[int, ...]:
    return tuple(np.r_[u.reshape(-1), v.reshape(-1), w.reshape(-1)].astype(int).tolist())


def commitment_hash(candidate: MasterCandidate) -> str:
    values = np.r_[candidate.u.ravel(), candidate.v.ravel(), candidate.w.ravel()]
    return hashlib.sha256(values.astype(np.uint8).tobytes()).hexdigest()


def make_candidate(
    data: UCData,
    scenarios: Sequence[Scenario],
    u: np.ndarray,
    source: str = "quantum",
) -> MasterCandidate:
    """Rebuild v/w from u and check all discrete constraints."""

    u = np.asarray(u, dtype=int).copy()
    if u.shape != (data.n_units, data.horizon):
        raise ValueError(f"u must have shape {(data.n_units, data.horizon)}")
    previous = np.column_stack((data.initial_on.astype(int), u[:, :-1]))
    v = np.maximum(u - previous, 0)
    w = np.maximum(previous - u, 0)
    return MasterCandidate(
        u=u,
        v=v,
        w=w,
        source=source,
    )


def _temporal_repair(data: UCData, u: np.ndarray) -> np.ndarray:
    """Repair transition, minimum up/down, and terminal-shutdown violations."""

    u = u.copy()
    for i in range(data.n_units):
        state = int(data.initial_on[i])
        duration = int(data.initial_duration[i])
        for t in range(data.horizon):
            minimum = int(data.min_up[i] if state else data.min_down[i])
            target = int(u[i, t])
            if target != state and (
                duration < minimum or (t == data.horizon - 1 and state == 1)
            ):
                target = state
            duration = duration + 1 if target == state else 1
            state = target
            u[i, t] = state
    return u


def reserve_capacity(data: UCData, candidate: MasterCandidate, scenario: Scenario) -> np.ndarray:
    capacity = commitment_capacity(data, candidate.u, candidate.v, candidate.w)
    return capacity.sum(axis=0) - (1.0 + data.reserve_fraction) * scenario.load


def _reserve_deficit(
    data: UCData,
    scenarios: Sequence[Scenario],
    candidate: MasterCandidate,
) -> np.ndarray:
    """Return the worst reserve deficit over all scenarios, by period."""

    deficits = np.stack(
        [np.maximum(0.0, -reserve_capacity(data, candidate, scenario)) for scenario in scenarios]
    )
    return deficits.max(axis=0)


def repair_commitment(
    data: UCData,
    scenarios: Sequence[Scenario],
    u: np.ndarray,
) -> MasterCandidate | None:
    """Repair temporal rules, then add online suffixes for deficient reserves.

    The repair is deterministic and only enforces the discrete master
    constraints. Continuous feasibility remains Gurobi's responsibility.
    """

    values = np.asarray(u, dtype=float)
    if values.shape != (data.n_units, data.horizon) or not np.all(np.isfinite(values)):
        return None
    binary = np.clip(np.rint(values), 0, 1).astype(int)
    candidate = make_candidate(data, scenarios, _temporal_repair(data, binary), source="repair")

    for _ in range(data.n_units + 1):
        if commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
            return candidate
        deficit = _reserve_deficit(data, scenarios, candidate)
        if not np.any(deficit > 1e-6):
            return None
        t = int(np.flatnonzero(deficit > 1e-6)[0])
        choices = []
        for i in range(data.n_units):
            start = (
                0
                if data.initial_on[i]
                else max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
            )
            trial = candidate.u.copy()
            trial[i, start:] = 1
            if np.array_equal(trial, candidate.u):
                continue
            option = make_candidate(data, scenarios, trial)
            remaining = _reserve_deficit(data, scenarios, option)
            gain = float(deficit[t] - remaining[t])
            if gain > 1e-6 and np.all(remaining <= deficit + 1e-6):
                fixed = float(
                    np.sum(data.cost_c[:, None] * option.u)
                    + np.sum(data.startup_cost[:, None] * option.v)
                    + np.sum(data.shutdown_cost[:, None] * option.w)
                )
                choices.append((gain, -fixed, -i, option))
        if not choices:
            return None
        candidate = max(choices, key=lambda item: item[:3])[3]
    return None
