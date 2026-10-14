"""Evidence and raw-candidate processing for the paper-style SA experiment."""
import hashlib
from time import perf_counter

import numpy as np

from .validation import pack


STREAMS = ('paper_states', 'paper_cuts', 'paper_bounds', 'paper_candidates', 'paper_coordination')


def context_id(b, outer, block):
    digest = hashlib.sha256(np.asarray(b.obj, '<f8').tobytes())
    digest.update(np.asarray([b.constant], '<f8').tobytes())
    digest.update(np.asarray(b.cols, '<i8').tobytes())
    return f'{outer}:{block}:{digest.hexdigest()}'


def record_state(whole, prices, outer):
    active = np.flatnonzero(prices)
    whole.diagnostics.setdefault('paper_states', []).append(dict(outer=outer,
        indices=active.tolist(), prices=np.asarray(prices)[active].tolist(),
        elapsed_seconds=perf_counter()-whole.started))


def record_cut(search, cut, x, source):
    identity = (bool(cut.optimality), float(cut.alpha), np.asarray(cut.beta, '<f8').tobytes())
    if identity in search.cut_keys:
        return False
    search.cut_keys.add(identity)
    if not cut.certified or not hasattr(cut, 'proof'):
        raise RuntimeError('Missing local cut proof')
    search.whole.diagnostics.setdefault('paper_cuts', []).append(dict(
        context=search.context, outer=search.outer, block=search.index, ordinal=len(search.cuts)+1,
        optimality=cut.optimality, alpha=cut.alpha, beta=cut.beta.tolist(), proof=cut.proof,
        x=np.asarray(x, np.uint8).tolist(), source=source,
        elapsed_seconds=perf_counter()-search.whole.started))
    return True


def record_bound(search, m):
    if m.bound_certificate is not None:
        search.whole.diagnostics.setdefault('paper_bounds', []).append(dict(
            context=search.context, outer=search.outer, block=search.index, cuts=len(search.cuts),
            constant=search.b.constant, **m.bound_certificate,
            elapsed_seconds=perf_counter()-search.whole.started))


def decode(q, text):
    bits = np.fromiter((int(v) for v in text), dtype=np.uint8)
    if len(bits) != q.Q.shape[0] or not np.isin(bits, [0, 1]).all():
        raise ValueError('Invalid complete SA bit string')
    binary = q.fixed.copy()
    binary[q.free] = bits[:len(q.free)]
    theta = q.theta_lower + bits[q.theta_ids] @ q.theta_weights
    return np.r_[binary, theta]


def sampling_evidence(search, m, q, sampling):
    residuals, feasible, theta = [], [], []
    for text in sampling.get('raw_bits', []):
        x = decode(q, text)
        residual = m.A @ x-m.rhs
        residual[m.eq] = abs(residual[m.eq])
        maximum = float(residual.max(initial=0))
        residuals.append(maximum)
        feasible.append(maximum <= 1e-7)
        theta.append(float(x[-1]))
    sampling.update(context=search.context, cut_count=len(search.cuts),
        native_master_residuals=residuals, native_master_feasible=feasible,
        theta_values=theta, fixed_indices=np.flatnonzero(~np.isnan(q.fixed)).tolist(),
        fixed_values=q.fixed[~np.isnan(q.fixed)].tolist(), free_indices=q.free.tolist(),
        theta_ids=q.theta_ids.tolist(), theta_weights=q.theta_weights.tolist(),
        theta_lower=q.theta_lower, removed_rows=q.removed_rows,
        max_coefficient=float(abs(q.Q).max()), nonzeros=int(q.Q.nnz))


def process_raw_batch(search, proposals, baseline_u, inner):
    """All untouched local evaluations precede any whole-day repair in this batch."""
    from .benders import discrete_feasible
    whole, b = search.whole, search.b
    unique = {}
    for k, x in enumerate(proposals):
        unique.setdefault(np.asarray(x, np.uint8).tobytes(), (k, x))
    local_results = {}
    for key, (_, x) in unique.items():
        if whole.remaining() <= 0:
            return
        local_results[key] = search.evaluate(x, 'sa_raw_local') if discrete_feasible(b, x) else None
    baseline = pack(whole.d, baseline_u)
    for key, (k, x) in unique.items():
        if whole.remaining() <= 0:
            return
        original = baseline.copy()
        original[b.cols[b.binary]] = x
        point = np.zeros(len(b.obj)); point[b.binary] = x
        raw_u = baseline_u.copy(); raw_u[:, b.times] = point[b.u]
        record = dict(proposal_id=f'sa:{whole.diagnostics["sa_calls"]}:{k}',
            outer=search.outer, block=search.index, round=inner, context=search.context,
            local_evaluated=local_results[key] is not None,
            local_feasible=bool(local_results[key] is not None and local_results[key].feasible))
        raw_record = dict(record)
        raw_ok = whole.accept(original, 'sa_raw', raw_record)
        repaired = whole.prepare(raw_u, 'sa_repaired', record)
        if repaired is not None and whole.remaining() > 0:
            record['hamming_uvw'] = int(np.count_nonzero(original != repaired))
            record['outside_block_hamming_u'] = int(np.count_nonzero(
                np.asarray(record['repaired_u'])[:, np.setdiff1d(np.arange(whole.d.horizon), b.times)]
                != baseline_u[:, np.setdiff1d(np.arange(whole.d.horizon), b.times)]))
            # Repair never supplies a local Benders point or a local cut.
            whole.accept(repaired, 'sa_repaired', record)
        whole.diagnostics.setdefault('paper_candidates', []).append(dict(record,
            raw_full_feasible=raw_ok, raw_incumbent_decrease=raw_record.get('incumbent_decrease', 0.),
            elapsed_seconds=perf_counter()-whole.started))
    whole.emit('paper_batch_complete')
