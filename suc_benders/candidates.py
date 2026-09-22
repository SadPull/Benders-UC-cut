from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .common import Scenario, commitment_is_feasible
from .data import UCData
from .master import MasterCandidate


def commitment_key(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[int, ...]:
    return tuple(np.r_[u.reshape(-1), v.reshape(-1), w.reshape(-1)].astype(int).tolist())


def make_candidate(
    data: UCData,
    scenarios: Sequence[Scenario],
    u: np.ndarray,
    theta: float | None = None,
    source: str = "quantum",
    move: dict | None = None,
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
        theta=theta,
        source=source,
        move=move,
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
    capacity = (
        data.pmax[:, None] * candidate.u
        - (data.pmax - data.pstart)[:, None] * candidate.v
    )
    capacity[:, :-1] -= (data.pmax - data.pshut)[:, None] * candidate.w[:, 1:]
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


def perturb_commitment(
    data: UCData,
    scenarios: Sequence[Scenario],
    base: MasterCandidate,
    *,
    limit: int = 32,
    window_lengths: Sequence[int] = (2, 3, 4),
    group_size: int = 2,
    seed: int | None = None,
) -> list[MasterCandidate]:
    """Generate feasible single-period, window, grouped, and replacement neighbors."""

    result: list[MasterCandidate] = []
    seen = {commitment_key(base.u, base.v, base.w)}
    rng = np.random.default_rng(seed)
    units = list(np.argsort(-(data.pmax - data.cost_b)))
    if seed is not None:
        rng.shuffle(units)

    def add(trial: np.ndarray, move: dict) -> bool:
        candidate = make_candidate(data, scenarios, trial, source="local", move=move)
        if not commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
            return False
        key = commitment_key(candidate.u, candidate.v, candidate.w)
        if key in seen:
            return False
        seen.add(key)
        result.append(candidate)
        return len(result) >= limit

    # Single-period moves are the cheapest source of local diversity.
    for t in range(data.horizon):
        for i in units:
            trial = base.u.copy()
            trial[i, t] = 1 - trial[i, t]
            if add(trial, {"move": "single", "unit": int(i), "period": int(t), "width": 1}):
                return result

    # Contiguous commitment-window moves.
    for width in window_lengths:
        width = int(width)
        if width <= 1:
            continue
        for start in range(max(0, data.horizon - width + 1)):
            for i in units:
                trial = base.u.copy()
                trial[i, start : start + width] = 1 - trial[i, start : start + width]
                if add(trial, {"move": "window", "unit": int(i), "period": int(start), "width": int(width)}):
                    return result

    # Grouped startup/shutdown and one-on-one replacement moves.
    group_size = max(2, min(int(group_size), data.n_units))
    groups = [units[j : j + group_size] for j in range(0, len(units), group_size)]
    groups = [group for group in groups if len(group) == group_size]
    for start in range(data.horizon):
        for group in groups:
            trial = base.u.copy()
            for i in group:
                trial[i, start] = 1 - trial[i, start]
            if add(trial, {"move": "group", "unit": int(group[0]), "period": int(start), "width": int(group_size)}):
                return result
        for on_unit in units:
            for off_unit in units:
                if on_unit == off_unit:
                    continue
                trial = base.u.copy()
                trial[on_unit, start] = 1
                trial[off_unit, start] = 0
                if add(trial, {"move": "replace", "unit": int(on_unit), "period": int(start), "width": 1, "off_unit": int(off_unit)}):
                    return result
    return result
