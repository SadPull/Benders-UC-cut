"""Author-aligned, single-bus SMS model; production UC conventions stay separate."""
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse


def enabled(d):
    return getattr(d, "model_profile", "") == "azema_sms"


def initial_forced_hours(d):
    # Author thermal_unit_commit_constraints also forces t=0,...,ceil((p0-pmin)/RD).
    return np.maximum(0, np.ceil((d.initial_output-d.pmin)/d.ramp_down)).astype(int)


def model(d, scenarios):
    n, T, S = d.n_units, d.horizon, len(scenarios)
    N = n*T
    u = np.arange(N).reshape((n, T), order="F")
    p = np.arange(3*N, (3+S)*N).reshape((n, T, S), order="F")
    shed = np.arange((3+S)*N, (3+S)*N+T*S).reshape((T, S), order="F")
    ix = NS(u=u, v=u+N, w=u+2*N, p=p, r=None, shed=shed, curtail=shed+T*S)
    nv = (3+S)*N+2*T*S
    m = NS(obj=np.zeros(nv), q=np.zeros(nv), lb=np.zeros(nv), ub=np.ones(nv), ix=ix,
           binary=np.arange(3*N), time=np.r_[np.tile(np.repeat(np.arange(T), n), 3+S),
                                            np.tile(np.arange(T), 2*S)],
           d=d, scenarios=scenarios, constant=0.)
    m.obj[ix.u], m.obj[ix.v], m.obj[ix.w] = d.cost_c[:, None], d.startup_cost[:, None], d.shutdown_cost[:, None]
    for k, s in enumerate(scenarios):
        m.obj[ix.p[:, :, k]] = s.probability*d.cost_b[:, None]
        m.ub[ix.p[:, :, k]] = d.pmax[:, None]
        m.obj[ix.shed[:, k]] = s.probability*d.shedding_cost
        m.obj[ix.curtail[:, k]] = s.probability*d.curtailment_cost
        # Positive penalties imply an optimum never sheds and curtails simultaneously.
        # These bounds retain the optimum for every commitment and its LP relaxation.
        m.ub[ix.shed[:, k]] = np.maximum(0., s.load)
        m.ub[ix.curtail[:, k]] = np.maximum(0., d.pmax.sum()-s.load)
    I, J, V, rhs, eq, labels = [], [], [], [], [], []

    def add(js, values, bound=0., equality=False, label=""):
        js, values = np.atleast_1d(js), np.atleast_1d(values)
        keep = values != 0
        I.extend([len(rhs)]*int(keep.sum())); J.extend(js[keep]); V.extend(values[keep])
        rhs.append(bound); eq.append(equality); labels.append(label)

    forced = initial_forced_hours(d)
    for t in range(T):
        for i in range(n):
            u, v, w = ix.u[i, t], ix.v[i, t], ix.w[i, t]
            if t == 0:
                add([u, v, w], [1, -1, 1], d.initial_on[i], True, "transition")
            else:
                add([u, v, w, ix.u[i, t-1]], [1, -1, 1, -1], 0., True, "transition")
            add([v, u], [1, -1], label="startup_on")
            add([w, u], [1, 1], 1., label="shutdown_off")
            up = ix.v[i, max(0, t-int(d.min_up[i])+1):t+1]
            down = ix.w[i, max(0, t-int(d.min_down[i])+1):t+1]
            up_initial = int(d.initial_on[i] and d.initial_duration[i] > 0
                             and t < d.min_up[i]-d.initial_duration[i])
            down_initial = int(not d.initial_on[i] and t < d.min_down[i]-d.initial_duration[i])
            add([*up, u], [*np.ones(len(up)), -1], -up_initial, label="min_up")
            add([*down, u], [*np.ones(len(down)), 1], 1-down_initial, label="min_down")
            if t < forced[i]:
                add(u, 1., 1., True, "initial_power")
            pm, pu, ru, rd = d.pmin[i], d.pmax[i], d.ramp_up[i], d.ramp_down[i]
            for k in range(S):
                p = ix.p[i, t, k]
                add([u, p], [pm, -1], label="pmin")
                add([p, u], [1, -pu], label="pmax")
                if t == 0:
                    add([p, v, u], [1, ru, -pm-ru], d.initial_output[i]-pm*d.initial_on[i], label="ramp_up")
                    add([p, w, u], [-1, rd, pm], (pm+rd)*d.initial_on[i]-d.initial_output[i], label="ramp_down")
                else:
                    pp, uprev = ix.p[i, t-1, k], ix.u[i, t-1]
                    add([p, pp, v, u, uprev], [1, -1, ru, -pm-ru, pm], label="ramp_up")
                    add([pp, p, w, uprev, u], [1, -1, rd, -pm-rd, pm], label="ramp_down")
        for k, s in enumerate(scenarios):
            add([*ix.p[:, t, k], ix.shed[t, k], ix.curtail[t, k]],
                [*np.ones(n), 1, -1], s.load[t], True, "balance")
    m.A = sparse.coo_matrix((V, (I, J)), shape=(len(rhs), nv)).tocsr()
    m.rhs, m.eq, m.labels = np.asarray(rhs, float), np.asarray(eq, bool), labels
    return m


def repair(d, u):
    u = np.clip(np.rint(u), 0, 1).copy()
    forced = initial_forced_hours(d)
    for i in range(d.n_units):
        state, duration = d.initial_on[i], d.initial_duration[i]
        if state and duration == 0:  # Author's initial minimum-up term uses signed duration > 0.
            duration = d.min_up[i]
        for t in range(d.horizon):
            minimum = d.min_up[i] if state else d.min_down[i]
            target = 1. if t < forced[i] else u[i, t]
            if target != state and duration < minimum:
                target = state
            duration = duration+1 if target == state else 1
            state = target; u[i, t] = state
    return u


def validate(d, scenarios, result):
    # Direct physical equations, independent of the sparse model and its row labels.
    u, v, w = (np.asarray(getattr(result, name)) for name in ("u", "v", "w"))
    shape = (d.n_units, d.horizon)
    if any(a.shape != shape or not np.isin(a, [0, 1]).all() for a in (u, v, w)):
        raise ValueError("Invalid SMS binary schedule")
    prior = np.column_stack([d.initial_on, u[:, :-1]])
    if not np.array_equal(v, np.maximum(u-prior, 0)) or not np.array_equal(w, np.maximum(prior-u, 0)):
        raise ValueError("SMS transition mismatch")
    for i in range(d.n_units):
        state, duration = d.initial_on[i], d.initial_duration[i]
        if state and duration == 0:
            duration = d.min_up[i]
        for t in range(d.horizon):
            if u[i, t] != state:
                if duration < (d.min_up[i] if state else d.min_down[i]):
                    raise ValueError("SMS minimum up/down time violated")
                state, duration = u[i, t], 1
            else:
                duration += 1
            if d.initial_output[i]-d.pmin[i] > t*d.ramp_down[i]+1e-9 and u[i, t] == 0:
                raise ValueError("SMS initial power shutdown restriction violated")
    if len(result.scenarios) != len(scenarios):
        raise ValueError("Missing SMS scenarios")
    maximum, fuel, penalty, shed_mwh, curtail_mwh = 0., 0., 0., 0., 0.
    for s, z in zip(scenarios, result.scenarios):
        p, shed, curtail = np.asarray(z.p), np.asarray(z.shed), np.asarray(z.curtail)
        if (z.name != s.name or p.shape != shape or shed.shape != (d.horizon,)
                or curtail.shape != (d.horizon,) or not all(np.isfinite(a).all() for a in (p, shed, curtail))):
            raise ValueError("Invalid SMS dispatch")
        previous = np.column_stack([d.initial_output, p[:, :-1]])
        up_limit = -d.ramp_up[:, None]*v+(d.pmin+d.ramp_up)[:, None]*u-d.pmin[:, None]*prior
        down_limit = -d.ramp_down[:, None]*w+(d.pmin+d.ramp_down)[:, None]*prior-d.pmin[:, None]*u
        residuals = [np.abs(p.sum(axis=0)+shed-curtail-s.load), d.pmin[:, None]*u-p,
                     p-d.pmax[:, None]*u, p-previous-up_limit, previous-p-down_limit, -p, -shed, -curtail]
        maximum = max(maximum, *(float(a.max(initial=0.)) for a in residuals))
        fuel_cost = float(np.sum(d.cost_b[:, None]*p))
        penalty_cost = float(d.shedding_cost*shed.sum()+d.curtailment_cost*curtail.sum())
        if abs(fuel_cost-z.objective) > 1e-4 or abs(penalty_cost-z.penalty_cost) > 1e-4:
            raise ValueError("SMS scenario cost mismatch")
        fuel += s.probability*fuel_cost; penalty += s.probability*penalty_cost
        shed_mwh += s.probability*float(shed.sum()); curtail_mwh += s.probability*float(curtail.sum())
    fixed = float(np.sum(d.cost_c[:, None]*u+d.startup_cost[:, None]*v+d.shutdown_cost[:, None]*w))
    error = abs(fixed+fuel+penalty-result.objective)
    if not np.isfinite(result.objective) or maximum > 1e-5 or error > 1e-4:
        raise ValueError("SMS physical or total cost validation failed")
    return dict(feasible=True, max_residual_mw=maximum, cost_residual=error,
                expected_fuel_cost=fuel, expected_penalty_cost=penalty,
                expected_shed_mwh=shed_mwh, expected_curtail_mwh=curtail_mwh,
                zero_shedding=shed_mwh <= 1e-7)
