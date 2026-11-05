"""Whole-day candidate repair, dispatch, validation and measured search statistics."""
import hashlib
from time import perf_counter

import numpy as np

from .benders import discrete_feasible
from .feedback import _cache_key
from .validation import capacity, commitment, commitment_hash, fixed_cost, pack, repair, reachable_shortfall, unpack


def prepare_candidate(whole, raw, source, record):
    """Repair a whole-day u proposal before full-day dispatch and validation."""
    values = np.asarray(raw, dtype='<f8')
    known = np.isfinite(whole.fixed_u)
    record.update(origin=source, raw_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                  raw_fixed_bit_violations=int(np.count_nonzero(values[known] != whole.fixed_u[known]))
                  if values.shape == whole.fixed_u.shape else 0)
    whole.diagnostics["proposals"].append(record)
    if whole.remaining() <= 0:
        record["outcome"] = "deadline"
        return None
    raw = np.asarray(raw, float)
    if raw.shape != (whole.d.n_units, whole.d.horizon) or not np.isfinite(raw).all():
        record["outcome"] = "invalid_matrix"
        return None
    raw_x = pack(whole.d, raw)
    z = commitment(whole.d, raw_x)
    required = (1 + whole.d.reserve_fraction) * np.max([s.load for s in whole.scenarios], axis=0)
    record["raw_discrete_feasible"] = bool(discrete_feasible(whole.full, raw_x)
        and np.array_equal(raw_x[np.isfinite(whole.fixed)], whole.fixed[np.isfinite(whole.fixed)])
        and (capacity(whole.d, z.u, z.v, z.w).sum(axis=0) >= required - 1e-6).all())
    phase = perf_counter()
    x = repair(whole.d, whole.scenarios, raw)
    whole.timings["repair"] += perf_counter() - phase
    if whole.remaining() <= 0:
        record["outcome"] = "deadline"
        return None
    if x is None:
        record["outcome"] = "repair_failed"
        return None
    z = commitment(whole.d, x)
    feasible = bool(discrete_feasible(whole.full, x)
        and (capacity(whole.d, z.u, z.v, z.w).sum(axis=0) >= required - 1e-6).all()
        and (np.sum(whole.d.pmin[:, None]*z.u, axis=0)
             <= np.min([s.load for s in whole.scenarios], axis=0) + 1e-6).all())
    record.update(commitment_id=commitment_hash(z),
                  hamming_u=int(np.count_nonzero(z.u != raw)), repaired_discrete_feasible=feasible,
                  duplicate=_cache_key(x) in whole.cache, outcome="repaired")
    if not feasible:
        record["outcome"] = "repair_discrete_infeasible"
        return None
    # 计入启停和爬坡链后可达出力仍低于负荷的方案必然调度不可行，跳过调度。
    if reachable_shortfall(whole.d, whole.scenarios, z.u):
        record["outcome"] = "reachable_infeasible"
        return None
    return x


def evaluate_candidate(whole, x, source, record=None, *, dispatch, validate):
    if x is None:
        if record is not None:
            record.setdefault("outcome", "no_candidate")
        return False
    if whole.remaining() <= 0:
        if record is not None:
            record["outcome"] = "deadline"
        return False
    key = _cache_key(x)
    if key in whole.cache:
        if record is not None:
            record.update(outcome="duplicate", duplicate=True, cache_hit=True, incumbent_decrease=0.)
        return whole.cache[key]["feasible"]
    phase = perf_counter()
    r = dispatch(whole.full, x, whole.remaining(), certified=True)
    whole.timings["subproblems"] += perf_counter() - phase
    if whole.remaining() <= 0 or r.status == "TIME_LIMIT":
        if record is not None:
            record["outcome"] = "dispatch_timeout"
        return False
    if r.status not in ("OPTIMAL", "INFEASIBLE", "DISCRETE_INFEASIBLE"):
        # SUBOPTIMAL 等数值状态只拒绝该候选且不缓存，求解继续。
        whole.diagnostics["dispatch_failures"] = whole.diagnostics.get("dispatch_failures", 0) + 1
        if record is not None:
            record["outcome"] = "dispatch_failed"
        return False
    before = whole.result.objective
    # plan 先是 u/v/w；可行时换成含各场景调度的完整解，哈希与费用都取自当前的 plan。
    plan = commitment(whole.d, x)
    binary_cost = fixed_cost(whole.d, plan.u, plan.w)
    report, scenario_costs = None, {}
    if r.feasible:
        plan = unpack(whole.m, r.point)
        phase = perf_counter()
        report = validate(whole.d, whole.scenarios, plan)
        whole.timings["validation"] += perf_counter() - phase
        scenario_costs = {s.name: s.objective for s in plan.scenarios}
    if whole.remaining() <= 0:
        if record is not None:
            record["outcome"] = "validation_deadline"
        return False
    improved = r.feasible and plan.objective < before
    # 先保护即将采用的新割；上限为 1 时也不能淘汰新 incumbent 的反馈。
    retained = r.cut is None or whole.add_cut(r.cut, key, protected=r.cut if improved else None)
    identity = commitment_hash(plan)
    reevaluated = identity in whole.evaluated_commitments
    entry = dict(source=source, feasible=r.feasible, dispatch_status=r.status,
                 objective=plan.objective if r.feasible else None,
                 fixed_cost=binary_cost, expected_fuel_cost=plan.objective-binary_cost if r.feasible else None,
                 scenario_costs=scenario_costs, commitment_hash=identity, reevaluated=reevaluated,
                 incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                 elapsed_seconds=perf_counter()-whole.started, sa_calls=whole.diagnostics["sa_calls"])
    if record is not None:
        entry["proposal_id"] = record.get("proposal_id")
        record["incumbent_decrease"] = float(before-plan.objective) if improved and np.isfinite(before) else 0.
    if retained:
        whole.cache[key] = entry
    whole.evaluated_commitments.add(identity)
    whole.diagnostics['re_evaluations'] += int(reevaluated)
    whole.diagnostics["evaluations"].append(entry)
    if record is not None:
        record.update(outcome="accepted" if r.feasible else "infeasible", feasible=r.feasible,
                      continuous_dispatch_completed=True, validation_completed=r.feasible)
    if improved:
        whole._adopt_incumbent(r, plan, report)
    whole.emit("evaluation")
    return r.feasible


def candidate_statistics(whole):
    """Count whole-day proposals and first validations, never block feasibility."""
    statistics = {}
    for source in ('sa_raw', 'sa'):
        proposals = [r for r in whole.diagnostics['proposals'] if r['origin'] == source]
        evaluations = [r for r in whole.diagnostics['evaluations'] if r['source'] == source]
        raw = {r['raw_sha256'] for r in proposals}
        repaired = {r['commitment_id'] for r in proposals if r.get('repaired_discrete_feasible')}
        outcomes = {}
        for r in proposals:
            outcome = r['outcome']; outcomes[outcome] = outcomes.get(outcome, 0)+1
        statistics[source] = dict(proposals=len(proposals), raw_unique=len(raw), repaired_unique=len(repaired),
            raw_duplicate_rate=1-len(raw)/len(proposals) if proposals else 0.,
            repaired_duplicate_rate=1-len(repaired)/sum(bool(r.get('repaired_discrete_feasible')) for r in proposals)
                if repaired else 0.,
            raw_discrete_violations=sum(r.get('raw_discrete_feasible') is False for r in proposals),
            raw_fixed_bit_violations=sum(r['raw_fixed_bit_violations'] for r in proposals),
            repair_bits=sum(r.get('hamming_u', 0) for r in proposals),
            duplicate_rate=outcomes.get('duplicate', 0)/len(proposals) if proposals else 0.,
            outcomes=outcomes,
            new_feasible=sum(r['feasible'] and not r['reevaluated'] for r in evaluations),
            incumbent_decrease=sum(r['incumbent_before']-r['incumbent'] for r in evaluations
                                   if r['improved'] and np.isfinite(r['incumbent_before'])))
    statistics['new_feasible'] = sum(row['new_feasible'] for row in statistics.values())
    return statistics
