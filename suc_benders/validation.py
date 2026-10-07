"""Temporal/capacity repair and independent physical/cost validation."""
from types import SimpleNamespace as NS
import hashlib

import numpy as np


def pack(d, u):
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    v, w = np.maximum(u - previous, 0), np.maximum(previous - u, 0)
    return np.concatenate([z.ravel(order="F") for z in (u, v, w)])


def commitment(d, x):
    # Each variable type occupies a contiguous, column-major unit/hour matrix.
    n = d.n_units * d.horizon
    return NS(**{name: np.asarray(x)[j*n:(j+1)*n].reshape((d.n_units, d.horizon), order="F").copy()
                 for j, name in enumerate(("u", "v", "w"))})


def commitment_hash(z):
    return hashlib.sha256(np.r_[z.u.ravel(), z.v.ravel(), z.w.ravel()].astype(np.uint8).tobytes()).hexdigest()


def capacity(d, u, v, w):
    startup = d.pmax[:, None] * u - (d.pmax - d.pstart)[:, None] * v
    shutdown = d.pmax[:, None] * u - (d.pmax - d.pshut)[:, None] * np.column_stack([w[:, 1:], np.zeros(d.n_units)])
    return np.minimum(startup, shutdown)


def repair(d, scenarios, u):
    u = np.asarray(u)
    if u.shape != (d.n_units, d.horizon) or not np.isfinite(u).all():
        return None
    u = np.clip(np.rint(u), 0, 1)
    for i in range(d.n_units):
        state, duration = d.initial_on[i], d.initial_duration[i]
        for t in range(d.horizon):
            minimum = d.min_up[i] if state else d.min_down[i]
            target = u[i, t]
            if target != state and (duration < minimum or (t == d.horizon - 1 and state == 1)):
                target = state
            duration = duration + 1 if target == state else 1
            state = target; u[i, t] = state
    required = (1 + d.reserve_fraction) * np.max([s.load for s in scenarios], axis=0)
    cost_vector = np.concatenate([np.tile(c, d.horizon) for c in (d.cost_c, d.startup_cost, d.shutdown_cost)])

    def evaluate(on):
        x = pack(d, on)
        n = d.n_units * d.horizon
        v, w = x[n:2 * n].reshape(on.shape, order="F"), x[2 * n:].reshape(on.shape, order="F")
        return x, np.maximum(0, required - capacity(d, on, v, w).sum(axis=0))

    for _ in range(d.n_units + 1):
        candidate, deficit = evaluate(u)
        if (deficit <= 1e-6).all():
            return candidate
        t = np.flatnonzero(deficit > 1e-6)[0]
        choices = []
        for i in range(d.n_units):
            start = 0 if d.initial_on[i] else max(0, int(d.min_down[i] - d.initial_duration[i]))
            if start >= d.horizon:
                continue
            trial = u.copy(); trial[i, start:] = 1
            x, remaining = evaluate(trial)
            gain, cost = deficit[t] - remaining[t], cost_vector @ x
            if (remaining <= deficit + 1e-6).all() and gain > 1e-6:
                choices.append((gain, -cost, -i, trial))
        if not choices:
            return None
        u = max(choices, key=lambda item: item[:3])[3]
    return None


def unpack(m, point):
    d = m.d
    result = NS(objective=float(m.obj @ point + m.q @ (point ** 2)), scenarios=[])
    for name in ("u", "v", "w"):
        setattr(result, name, point[getattr(m.ix, name)].copy())
    for k, s in enumerate(m.scenarios):
        p, r = point[m.ix.p[:, :, k]], point[m.ix.r[:, :, k]]
        result.scenarios.append(NS(name=s.name, probability=s.probability, p=p.copy(), r=r.copy(),
                                   objective=float(np.sum(d.cost_a[:, None] * p ** 2 + d.cost_b[:, None] * p))))
    return result


def validate(d, scenarios, result):
    if not np.isfinite(result.objective):
        raise ValueError("No finite objective")
    for name in ("u", "v", "w"):
        z = getattr(result, name)
        if z.shape != (d.n_units, d.horizon) or not np.isin(z, [0, 1]).all():
            raise ValueError("Invalid binary schedule")
    u, v, w = result.u, result.v, result.w
    previous = np.column_stack([d.initial_on, u[:, :-1]])
    if not np.array_equal(v, np.maximum(u - previous, 0)) or not np.array_equal(w, np.maximum(previous - u, 0)):
        raise ValueError("Startup/shutdown transition mismatch")
    if w[:, -1].any():
        raise ValueError("Terminal shutdown is forbidden")
    for i in range(d.n_units):
        state, duration = d.initial_on[i], d.initial_duration[i]
        for t in range(d.horizon):
            minimum = d.min_up[i] if state else d.min_down[i]
            if u[i, t] != state:
                if duration < minimum:
                    raise ValueError("Minimum up/down time violated")
                state, duration = u[i, t], 1
            else:
                duration += 1
    if len(result.scenarios) != len(scenarios):
        raise ValueError("Missing scenarios")
    maximum, fuel = 0., 0.
    for s, z in zip(scenarios, result.scenarios):
        p, r = z.p, z.r
        if (z.name != s.name or p.shape != u.shape or r.shape != u.shape
                or not np.isfinite(p).all() or not np.isfinite(r).all()):
            raise ValueError("Invalid dispatch")
        previous = np.column_stack([d.initial_output, p[:, :-1]])
        residuals = [np.abs(p.sum(axis=0) - s.load), d.reserve_fraction * s.load - r.sum(axis=0),
                     d.pmin[:, None] * u - p, p + r - capacity(d, u, v, w),
                     p + r - d.pmax[:, None] * u, p - previous - d.ramp_up[:, None],
                     previous - p - d.ramp_down[:, None], -p, -r]
        maximum = max(maximum, *(float(a.max()) for a in residuals))
        cost = float(np.sum(d.cost_a[:, None] * p ** 2 + d.cost_b[:, None] * p))
        if abs(cost - z.objective) > 1e-4:
            raise ValueError("Scenario fuel cost mismatch")
        fuel += s.probability * cost
    fixed = np.sum(d.cost_c[:, None] * u + d.startup_cost[:, None] * v + d.shutdown_cost[:, None] * w)
    error = float(abs(fixed + fuel - result.objective))
    if maximum > 1e-5 or error > 1e-4:
        raise ValueError("Physical or total cost validation failed")
    return {"feasible": True, "max_residual_mw": maximum, "cost_residual": error}
