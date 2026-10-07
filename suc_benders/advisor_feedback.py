"""Measured proposal feedback, generated without another model call."""
import numpy as np


def call_due(schedule, before, after, stalled, calls, patience=3):
    """Update SA stagnation; the caller resets it only after an actual call."""
    if schedule not in {"fixed", "stagnation", "stagnation_burst"}:
        raise ValueError("unknown LLM schedule")
    improvement = (before-after)/max(abs(before), 1e-12) if np.isfinite(before) else 0.
    stalled = 0 if improvement >= .0005 else stalled+1
    return schedule == "fixed" or (schedule == "stagnation" and calls == 0) or stalled >= patience, stalled


def changes(before, after):
    if before is None or after is None:
        return []
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape:
        return []
    rows = []
    for i in range(len(before)):
        for t in np.flatnonzero(before[i] != after[i]):
            old, new = float(before[i, t]), float(after[i, t])
            if not np.isfinite(old) or not np.isfinite(new):
                continue
            if rows and rows[-1]["unit"] == i+1 and rows[-1]["end_hour"] == t and rows[-1]["from"] == old and rows[-1]["to"] == new:
                rows[-1]["end_hour"] = int(t+1)
            else:
                rows.append(dict(unit=i+1, start_hour=int(t+1), end_hour=int(t+1), **{"from": old, "to": new}))
    return rows


def summarize(advice, baseline_u, baseline, evaluations, previous_ids):
    measured = {r["commitment_hash"]: r for r in evaluations}
    rows = []
    for p in advice.diagnostics.get("proposals", []):
        identity = p.get("commitment_id")
        r = measured.get(identity)
        cached = identity in previous_ids or p.get("outcome") == "duplicate"
        row = dict(proposal_id=p["proposal_id"], commitment_id=identity,
                   proposed_changes=changes(baseline_u, p.get("raw_u")),
                   repair_changes=changes(p.get("raw_u"), p.get("repaired_u")),
                   repair_bits=p.get("hamming_u"), outcome=p.get("outcome"),
                   duplicate=cached, cache_hit=bool(cached and r),
                   continuous_dispatch_completed=bool(r or p.get("local_dispatch_completed")),
                   local_feasible=p.get("local_feasible"),
                   validation_completed=r is not None, feasible=r["feasible"] if r else None,
                   repair_reason=None, infeasibility_reason=None,
                   baseline_cost=baseline, costs=None, cost_change=None,
                   refreshed_incumbent=bool(r and r["improved"] and not cached))
        if r and r["feasible"]:
            costs = dict(total=r["objective"], fixed=r["fixed_cost"], expected_fuel=r["expected_fuel_cost"])
            row["costs"] = costs
            if baseline is not None:
                row["cost_change"] = {k: costs[k]-baseline[k] for k in costs}
        rows.append(row)
    return dict(call_id=advice.diagnostics.get("call_id"), failed=advice.failed,
                message=advice.message, proposals=rows)
