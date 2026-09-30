from __future__ import annotations

from dataclasses import dataclass
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


def on_blocks(row: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] spans of consecutive ON hours."""

    blocks = []
    t = 0
    horizon = int(np.asarray(row).size)
    values = np.asarray(row, dtype=int)
    while t < horizon:
        if values[t]:
            start = t
            while t < horizon and values[t]:
                t += 1
            blocks.append((start, t - 1))
        else:
            t += 1
    return blocks


def _fixed_cost_of(data: UCData, candidate: MasterCandidate) -> float:
    return float(
        np.sum(data.cost_c[:, None] * candidate.u)
        + np.sum(data.startup_cost[:, None] * candidate.v)
        + np.sum(data.shutdown_cost[:, None] * candidate.w)
    )


def _trim_options(start: int, end: int) -> list[tuple[int, int, dict]]:
    """Removal and end-trim options for one ON-block, cheapest-to-apply first."""

    options: list[tuple[int, int, dict]] = [
        (start, end, {"move": "decommit_block", "period": int(start), "width": int(end - start + 1)})
    ]
    # Trimming the shoulders of a block is the move the exact solution uses to
    # drop peakers after the peak, so offer both ends explicitly.
    if end > start:
        options.append((end, end, {"move": "decommit_trim_end", "period": int(end), "width": 1}))
        options.append((start, start, {"move": "decommit_trim_start", "period": int(start), "width": 1}))
    if end - start >= 1:
        options.append((max(start, end - 1), end, {"move": "decommit_trim_end2", "period": int(max(start, end - 1)), "width": int(min(2, end - start + 1))}))
        options.append((start, min(end, start + 1), {"move": "decommit_trim_start2", "period": int(start), "width": int(min(2, end - start + 1))}))
    return options


def decommit_commitment(
    data: UCData,
    scenarios: Sequence[Scenario],
    base: MasterCandidate,
    *,
    limit: int = 8,
    gradient: dict[tuple[int, int], float] | None = None,
) -> list[MasterCandidate]:
    """Greedy decommitment ladder starting from an incumbent commitment.

    Each rung turns OFF hours that the incumbent currently keeps running, which
    is the direction the exact solution moves: the gap is dominated by fixed
    cost from over-committed peaking units.

    When ``gradient`` is supplied it maps ``(unit, period) -> estimated change
    in total cost from switching that unit-hour off`` (negative is good). The
    rung score is then the predicted total cost change of the whole move, i.e.
    the exact fixed-cost saving plus the summed fuel-cost estimate. Without a
    gradient the exact fixed-cost saving is used alone.

    Candidate selection uses only cheap discrete checks; no Gurobi subproblem is
    solved here, the caller evaluates each rung.
    """

    current = np.asarray(base.u, dtype=int).copy()
    ladder: list[MasterCandidate] = []
    seen = {commitment_key(base.u, base.v, base.w)}

    for _ in range(max(1, int(limit))):
        current_candidate = make_candidate(data, scenarios, current)
        base_fixed = _fixed_cost_of(data, current_candidate)
        best = None
        best_score: float | None = None
        for i in range(data.n_units):
            for start, end in on_blocks(current[i]):
                for t1, t2, move in _trim_options(start, end):
                    trial = current.copy()
                    trial[i, t1 : t2 + 1] = 0
                    option = make_candidate(
                        data,
                        scenarios,
                        trial,
                        source="decommit",
                        move={"unit": int(i), **move},
                    )
                    if not commitment_is_feasible(
                        data, scenarios, option.u, option.v, option.w
                    ):
                        continue
                    # Realised fixed-cost saving of this move.
                    saving = base_fixed - _fixed_cost_of(data, option)
                    if saving <= 1e-6:
                        continue
                    if gradient:
                        fuel_estimate = 0.0
                        for period in range(t1, t2 + 1):
                            cell = gradient.get((int(i), int(period)))
                            if cell is not None:
                                # The gradient already subtracts no-load cost;
                                # saving accounts for it for the whole move.
                                fuel_estimate += float(cell) + float(data.cost_c[i])
                        # Predicted change in total cost; lower is better.
                        score = -saving + fuel_estimate
                    else:
                        score = -saving
                    if best_score is None or score < best_score:
                        best_score = score
                        best = option
        if best is None:
            break
        key = commitment_key(best.u, best.v, best.w)
        current = best.u.copy()
        if key in seen:
            break
        seen.add(key)
        ladder.append(best)

    return ladder
