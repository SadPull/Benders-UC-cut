"""Replay continuous proofs and check saved paper-SA schedules without integer search."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from suc_benders.benders import master
from suc_benders.model import fingerprint, partition
from suc_benders.paper import context_id
from .artifacts import read_jsonl, save, scored_trace, sha256
from .rts24 import case, fixed_check, quadratic_dual


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def proof_prefixes(folder, m, offsets):
    states = read_jsonl(folder/'paper_states.jsonl')[:offsets.get('paper_states', 0)]
    blocks, contexts, lower = {}, {}, {}
    for state in states:
        outer = state['outer']
        if outer in blocks:
            raise ValueError('Duplicate pricing state')
        prices = np.zeros(len(m.rhs)); prices[state['indices']] = state['prices']
        blocks[outer], _ = partition(m, 3, prices)
        lower[outer] = []
        for j, b in enumerate(blocks[outer], 1):
            b.paper_sa = True
            contexts[context_id(b, outer, j)] = b
            lower[outer].append(float(np.minimum(b.obj*b.lb, b.obj*b.ub).sum()+b.constant))
    cuts = {}
    for row in read_jsonl(folder/'paper_cuts.jsonl')[:offsets.get('paper_cuts', 0)]:
        b = contexts[row['context']]
        if row['context'] != context_id(b, row['outer'], row['block']):
            raise ValueError('Cut pricing context mismatch')
        group = cuts.setdefault(row['context'], [])
        if row['ordinal'] != len(group)+1:
            raise ValueError('Missing or reordered local cut')
        verify_cut(b, row)
        group.append(row)
    initial = read(folder/'initial_bound.json')
    if initial['model_fingerprint'] != fingerprint(m) or initial['arrays_sha256'] != sha256(folder/'initial_prices.npz'):
        raise ValueError('Initial proof identity mismatch')
    initial_value = quadratic_dual(m, np.load(folder/'initial_prices.npz')['prices'])
    if initial['lower_bound'] > initial_value+1e-6:
        raise ValueError('Initial lower bound exceeds its quadratic dual')
    prefix = [initial['lower_bound']]
    for row in read_jsonl(folder/'paper_bounds.jsonl')[:offsets.get('paper_bounds', 0)]:
        b = contexts[row['context']]
        if row['constant'] != b.constant or row['context'] != context_id(b, row['outer'], row['block']):
            raise ValueError('Local bound pricing or constant mismatch')
        group = cuts.get(row['context'], [])[:row['cuts']]
        if len(group) != row['cuts'] or any(c['elapsed_seconds'] > row['elapsed_seconds'] for c in group):
            raise ValueError('Bound uses future or missing cuts')
        native, _ = master(b, [NS(**dict(c, beta=np.asarray(c['beta']))) for c in group])
        prices = np.zeros(len(native.rhs))
        indices = np.asarray(row['indices'], int)
        if len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= len(prices)).any():
            raise ValueError('Invalid master dual indices')
        prices[indices] = row['prices']
        value = quadratic_dual(native, prices)+b.constant
        if not np.isfinite(row['lower']) or row['lower'] > value+1e-6:
            raise ValueError('Local LP bound exceeds its continuous dual proof')
        values = lower[row['outer']]
        values[row['block']-1] = max(values[row['block']-1], row['lower'])
        prefix.append(max(prefix[-1], sum(values)))
    return prefix, dict(initial_dual_value=initial_value, local_cuts=sum(map(len, cuts.values())),
                        lp_proofs=len(prefix)-1, pricing_states=len(states))


def audit(folder, inputs, budget, output, remaining):
    folder, output = Path(folder), Path(output)
    rows = scored_trace(folder, budget); verify_trace(rows)
    if not rows:
        raise ValueError('No completed in-budget checkpoint')
    m = case(Path(inputs)); signature = fingerprint(m)
    final_state = read(folder/rows[-1]['checkpoint'])
    prefix, proof = proof_prefixes(folder, m, final_state['offsets'])
    calls = [r for r in read_jsonl(folder/'optimizer_calls.jsonl') if r['elapsed_seconds'] <= rows[-1]['elapsed_seconds']]
    starts = [r for r in calls if r['event'] == 'start']
    if not starts or any(r['integer_variables'] or r['threads'] != 1 for r in calls):
        raise ValueError('Formal Gurobi integer search or multithreading detected')
    schedules = {}
    for row in rows:
        state = read(folder/row['checkpoint'])
        if state['model_fingerprint'] != signature or row['llm_calls']:
            raise ValueError('Scored model changed or LLM was used')
        bound_index = state['offsets'].get('paper_bounds', 0)
        if row['lower_bound'] > prefix[bound_index]+1e-5:
            raise ValueError('Global lower bound lacks a same-price sum of continuous proofs')
        schedules[np.asarray(state['u'], np.uint8).tobytes()] = (state['u'], row['objective'])
    checked = []
    for k, (u, objective) in enumerate(schedules.values()):
        if remaining() <= 0:
            raise TimeoutError('Schedule audit budget exhausted')
        value = fixed_check(case(Path(inputs)), u, objective, remaining())
        if not value['validation']['feasible']:
            raise ValueError('Audited schedule is infeasible')
        path = output/f'incumbent-{k+1:03d}.json'
        save(path, dict(u=u, **value))
        checked.append(dict(artifact=path.name, objective=value['objective'], validation=value['validation']))
    final_key = np.asarray(final_state['u'], np.uint8).tobytes()
    final_index = list(schedules).index(final_key)
    result = dict(passed=True, final=rows[-1], model_fingerprint=signature, **proof,
                  gurobi_integer_calls=0, gurobi_continuous_calls=len(starts), incumbents=checked,
                  final_solution=checked[final_index]['artifact'])
    save(output/'audit.json', result)
    return result


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


def verify_trace(rows):
    last_time, last_lower, last_upper = -np.inf, -np.inf, np.inf
    for row in rows:
        t, lb, ub = row['elapsed_seconds'], row['lower_bound'], row['objective']
        if not np.isfinite([t,lb,ub]).all() or t < last_time or lb > ub+1e-5:
            raise ValueError('Invalid timestamp or crossed bounds')
        if lb < last_lower-1e-5 or ub > last_upper+1e-5:
            raise ValueError('Nonmonotone certified trace')
        last_time, last_lower, last_upper = t, lb, ub
