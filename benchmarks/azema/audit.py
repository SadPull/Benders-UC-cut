"""Recompute physical dispatch with the author model and verify every saved cut algebraically."""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from suc_benders.data import load_data
from suc_benders.hourly_bounds import HourlyBounds
from suc_benders.model import model, partition, fingerprint
from suc_benders.validation import commitment, commitment_hash, pack, validate
from .artifacts import read_jsonl, save, scored_trace, sha256


def verify_cut(b, cut):
    """Weak duality over finite recourse bounds; no solver or sampled commitments needed."""
    X, Y = b.binary, b.continuous
    rows = np.flatnonzero(b.A[:, Y].getnnz(axis=1))
    A, B, rhs, eq = b.A[rows][:, Y], b.A[rows][:, X], b.rhs[rows], b.eq[rows]
    proof = cut['proof']
    if cut['optimality']:
        if proof['kind'] != 'bounded_lagrangian':
            raise ValueError('Wrong optimality proof')
    else:
        if proof['kind'] != 'bounded_farkas':
            raise ValueError('Wrong feasibility proof')
        E = np.flatnonzero(eq)
        A, B = sparse.vstack([A, -A[E]], format='csr'), sparse.vstack([B, -B[E]], format='csr')
        rhs, eq = np.r_[rhs, -rhs[E]], np.zeros(len(rhs)+len(E), bool)
    prices = np.zeros(len(rhs))
    indices = np.asarray(proof['indices'], int)
    if (len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= len(rhs)).any()):
        raise ValueError('Invalid dual indices')
    prices[indices] = proof['prices']
    if not np.isfinite(prices).all() or (prices[~eq] < 0).any():
        raise ValueError('Invalid dual multiplier sign')
    c = np.asarray(A.T @ prices)+(b.obj[Y] if cut['optimality'] else 0)
    q = b.q[Y] if cut['optimality'] else np.zeros(len(Y))
    z = np.where(c < 0, b.ub[Y], b.lb[Y]).copy()
    curved = q > 0
    z[curved] = np.clip(-c[curved]/(2*q[curved]), b.lb[Y][curved], b.ub[Y][curved])
    alpha = float(c @ z+q @ z**2-rhs @ prices)
    beta = np.asarray(B.T @ prices)
    if (not np.isfinite(alpha) or not np.isfinite(cut['alpha'])
            or cut['alpha'] > alpha+1e-6 or not np.allclose(beta, cut['beta'], atol=1e-7, rtol=1e-12)):
        raise ValueError('Saved cut is not implied by its dual certificate')


def verify_author_cut(d, scenarios, cut):
    n, T = d.n_units, d.horizon
    beta, alpha = np.zeros(3*n*T), 0.
    proofs = cut['proof']
    if cut['optimality'] and sorted(p['scenario'] for p in proofs) != list(range(1,len(scenarios)+1)):
        raise ValueError('Author cut does not cover all scenarios exactly once')
    for proof in proofs:
        arrays = {}
        for key in ('mumin','mumax','muup','mudown','nu'):
            shape = (1,T) if key == 'nu' else (n,T)
            a = np.zeros(np.prod(shape))
            a[np.asarray(proof[key]['indices'], int)-1] = proof[key]['values']
            arrays[key] = a.reshape(shape, order='F')
        lo, hi, up, down, nu = (arrays[k] for k in ('mumin','mumax','muup','mudown','nu'))
        if any(not np.isfinite(a).all() for a in arrays.values()) or min(lo.min(),hi.min(),up.min(),down.min()) < -1e-8:
            raise ValueError('Invalid author dual multipliers')
        stationarity = lo-hi-up+down+nu
        stationarity[:,:-1] += up[:,1:]-down[:,1:]
        target = d.cost_b[:,None] if cut['optimality'] else np.zeros((n,1))
        limit = 700. if cut['optimality'] else 0.
        if np.max(abs(stationarity-target)) > 1e-6 or np.max(abs(nu)) > limit+1e-6:
            raise ValueError('Author cut violates dual stationarity or imbalance price bounds')
        previous = up*d.pmin[:,None]-down*(d.pmin+d.ramp_down)[:,None]
        u = lo*d.pmin[:,None]-hi*d.pmax[:,None]-up*(d.pmin+d.ramp_up)[:,None]+down*d.pmin[:,None]
        u[:,:-1] += previous[:,1:]
        v, w = up*d.ramp_up[:,None], down*d.ramp_down[:,None]
        beta += np.r_[u.ravel(order='F'),v.ravel(order='F'),w.ravel(order='F')]
        s = scenarios[proof['scenario']-1]
        alpha += float(nu @ s.load+(down[:,0]-up[:,0]) @ d.initial_output+previous[:,0] @ d.initial_on)
    alpha /= len(proofs); beta /= len(proofs)
    if abs(cut['alpha']-alpha) > 1e-5 or not np.allclose(cut['beta'],beta,atol=1e-7,rtol=1e-12):
        raise ValueError('Author cut coefficients differ from its saved duals')


def verify_trace(rows):
    last_time, last_lower, last_upper = -np.inf, -np.inf, np.inf
    for row in rows:
        t, lb, ub = row['elapsed_seconds'], row['lower_bound'], row['objective']
        if not np.isfinite([t,lb,ub]).all() or t < last_time or lb > ub+1e-5:
            raise ValueError('Invalid timestamp or crossed bounds')
        if lb < last_lower-1e-5 or ub > last_upper+1e-5:
            raise ValueError('Nonmonotone certified trace')
        last_time, last_lower, last_upper = t, lb, ub


def checked_author_schedule(d, scenarios, raw, u, expected):
    z = commitment(d,pack(d,np.asarray(u)))
    z.scenarios = []
    for data in raw['scenarios']:
        p, shed, curtail = (np.asarray(data[name],float) for name in ('p','shed','curtail'))
        fuel = float(np.sum(d.cost_b[:,None]*p))
        penalty = float(d.shedding_cost*shed.sum()+d.curtailment_cost*curtail.sum())
        z.scenarios.append(NS(**data, objective=fuel, penalty_cost=penalty))
    z.objective = raw['objective']
    checked = validate(d,scenarios,z)
    if abs(z.objective-expected) > 1e-4:
        raise ValueError('Independent all-scenario cost differs from the scored incumbent')
    return checked


def audit_run(job):
    from .preflight import author_check
    started = perf_counter()
    run, out, inputs = (Path(job[key]) for key in ('run_directory','directory','inputs'))
    rows = scored_trace(run,job['budget'])
    if not rows:
        raise ValueError('No in-budget checkpoint to audit')
    verify_trace(rows)
    d, scenarios = load_data(inputs/'units.json',inputs/'scenarios.json')
    units = json.loads((inputs/'units.json').read_text(encoding='utf-8'))
    scenario_json = json.loads((inputs/'scenarios.json').read_text(encoding='utf-8'))
    m = model(d,scenarios); full = partition(m,d.horizon)[0][0]; signature = fingerprint(m)
    hourly = HourlyBounds(full)
    cuts = read_jsonl(run/'certificate_cuts.jsonl')
    hour_cuts = read_jsonl(run/'hourly_certificate_cuts.jsonl')
    for cut in cuts:
        if cut['elapsed_seconds'] <= job['budget']:
            if job['arm'] == 'A':
                verify_author_cut(d,scenarios,cut)
            else:
                verify_cut(full,cut)
    for cut in hour_cuts:
        if cut['elapsed_seconds'] <= job['budget']:
            verify_cut(hourly.parts[cut['hour']-1].b,dict(cut,optimality=True))
    # Audit every distinct saved incumbent; trace entries share schedules by hash.
    schedules = {}
    common = json.loads((Path(job['common_directory'])/'common.json').read_text(encoding='utf-8'))
    native_history = read_jsonl(run/'certification_history.jsonl')
    for row in rows:
        state = json.loads((run/row['checkpoint']).read_text(encoding='utf-8'))
        if state['model_fingerprint'] != signature:
            raise ValueError('Checkpoint model fingerprint mismatch')
        if row['lower_bound'] > common['lower_bound']+1e-5:
            certificate = state.get('certificate') if job['arm'] != 'A' else state.get('bound_source')
            if not certificate:
                raise ValueError('Improved lower bound has no native master evidence')
            if job['arm'] == 'A':
                raw_bound = certificate.get('raw_bound')
                if (certificate.get('kind') != 'author_master' or raw_bound is None
                        or not np.isfinite(raw_bound) or abs(raw_bound-1e-5-row['lower_bound']) > 1e-5
                        or certificate['cuts'] > state['cuts']):
                    raise ValueError('Author lower bound has inconsistent native master evidence')
            else:
                offsets = state['offsets']
                valid = []
                for native in native_history[:offsets['certification_history']]:
                    if not native.get('bound_used'):
                        continue
                    if (native['status'] not in ('OPTIMAL', 'TIME_LIMIT', 'INTERRUPTED', 'USER_OBJ_LIMIT')
                            or native['cuts'] > offsets['certificate_cuts']
                            or native['hourly_cuts'] > offsets['hourly_certificate_cuts']
                            or not np.isfinite(native['master_bound'])):
                        raise ValueError('Native bound uses missing cuts or invalid solver evidence')
                    valid.append(native['master_bound'])
                if not valid or row['lower_bound'] > max(valid)+1e-5:
                    raise ValueError('Scored lower bound is not supported by a saved native master bound')
        key = commitment_hash(commitment(d,pack(d,np.asarray(state['u']))))
        if key not in schedules:
            remaining = job['time_limit']-(perf_counter()-started)
            if remaining <= 0:
                raise TimeoutError('Independent audit budget exhausted')
            reference = author_check(out/key, units, scenario_json, 'fixed', state['u'],
                                     seconds=min(120,remaining), all_scenarios=True)
            schedules[key] = dict(objective=row['objective'], **checked_author_schedule(
                d,scenarios,reference,state['u'],row['objective']))
            save(out/'incumbents.json',schedules)
    final = rows[-1]
    save(out/'audit.json',dict(passed=True, arm=job['arm'], final=final, incumbents=schedules,
        lower_bound_audit='native_Gurobi_bound_with_algebraically_verified_cuts',
        global_cuts=len(cuts), hourly_cuts=len(hour_cuts), model_fingerprint=signature,
        trace_sha256=sha256(run/'trace.jsonl'), elapsed_seconds=perf_counter()-started,
        scope='All in-budget cuts and every distinct scored incumbent; all 100 scenarios per incumbent'))
