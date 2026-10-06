"""Read-only reconstruction and post-search integer-bound certification."""
from __future__ import annotations

import copy
import math
import time

from .block_master_model import block_dual_floor, partition_rows, priced_blocks, solve_master
from .coordination import cutting_plane
from .paper_model import Cut, IndexedUC
from .pure_sa import interval_fixed_binary
from .sa_scale_experiment import events


def committed_events(path):
    pending, committed = {}, []
    for row in events(path):
        if row["kind"] == "event_ready":
            original = pending.pop(row["sequence"], None)
            if original is not None:
                if row["clock"] < original["clock"]:
                    raise ValueError("event committed before it was emitted")
                committed.append({**original, "emit_clock": original["clock"], "clock": row["clock"]})
        elif "sequence" in row:
            pending[row["sequence"]] = row
    return committed


def point_value(record):
    return {tuple(k): float(v) for k, v in record}


def snapshot(rows, cutoff):
    epochs, initial_prices = {}, None
    initial_lower, upper = -math.inf, math.inf
    for row in rows:
        if row["clock"] > cutoff:
            continue
        kind = row["kind"]
        if kind == "initial_dual_prices":
            initial_prices = list(row["multipliers"])
        elif kind == "bound_update" and row["outer"] == 0:
            initial_lower = max(initial_lower, row["reliable_lower"])
        elif kind == "multiplier_epoch":
            epochs[row["outer"]] = dict(outer=row["outer"], multipliers=list(row["multipliers"]),
                source_clock=row["clock"], blocks=[dict(**b, cuts=[], best=None) for b in row["blocks"]])
        elif kind in ("local_cut", "block_bound"):
            epoch = epochs.get(row["outer"])
            if epoch is None:
                raise ValueError("block record without multiplier epoch")
            block = epoch["blocks"][row["block"]]
            epoch["source_clock"] = row["clock"]
            if kind == "local_cut":
                block["cuts"].append(copy.deepcopy(row["cut"]))
                if row["feasible"] and (block["best"] is None or row["cost"] < block["best"]["cost"]):
                    block["best"] = dict(cost=row["cost"], point=copy.deepcopy(row["point"]),
                                         sources=list(row["sources"]))
            else:
                block["lower"] = max(block["lower"], row["lower"])
                block["converged"] = row["converged"]
        if kind in ("bound_update", "iteration", "approximate_outer"):
            bound = row.get("coordination_upper", row.get("upper") if kind == "approximate_outer" else None)
            if bound is not None and math.isfinite(bound):
                upper = min(upper, bound)
    return dict(cutoff_clock=cutoff, initial_lower=initial_lower,
                initial_local_prices=initial_prices, coordination_upper=upper,
                epochs=list(epochs.values()))


def verify_epoch(indexed, definition, view, epoch):
    blocks, cross = priced_blocks(indexed, definition["hours"], epoch["multipliers"])
    if len(blocks) != len(epoch["blocks"]):
        raise ValueError("snapshot block count mismatch")
    compact = definition["method"] in ("sa_compact", "sa_full")
    inexact = definition["method"] != "sa_strict"
    for block, saved in zip(blocks, epoch["blocks"]):
        if compact:
            block.fixed = interval_fixed_binary(block.integer_rows, block.fixed)
        if list(block.times) != saved["times"] or abs(block.constant-saved["constant"]) > 1e-5:
            raise ValueError("snapshot partition or constant mismatch")
        floor = block.objective_bounds(block.costs)[0]+block.constant
        if inexact:
            if view["initial_local_prices"] is None:
                raise ValueError("missing initial local dual prices")
            floor = max(floor, block_dual_floor(block, view["initial_local_prices"]))
        if not math.isfinite(saved["lower"]) or saved["lower"] < floor-1e-5:
            raise ValueError("invalid continuous block lower bound")
        cuts = [Cut(point_value(c["a"]), c["c"], c["feasible"]) for c in saved["cuts"]]
        if any(not math.isfinite(c.c) or any(not math.isfinite(v) for v in c.a.values()) for c in cuts):
            raise ValueError("nonfinite unquantized cut")
        if saved["best"] is not None:
            point = point_value(saved["best"]["point"])
            cutting_plane(block, cross, point)
            cost = block.constant + sum(block.costs[k]*point[k]+indexed.quadratic.get(k, 0.)*point[k]**2
                                        for k in block.keys)
            if abs(cost-saved["best"]["cost"]) > 1e-5:
                raise ValueError("local witness cost mismatch")
            if saved["lower"] > cost+1e-5:
                raise ValueError("continuous block lower exceeds local witness")
            recourse = cost-block.constant-sum(block.costs[k]*point[k] for k in block.bkeys)
            if any(c.value(point) > (recourse if c.feasible else 0.)+1e-5 for c in cuts):
                raise ValueError("cut excludes a saved feasible local witness")
        yield block, cuts, saved


def coalesce_epochs(indexed, hours, epochs):
    """Union saved cuts only when effective cross-block prices are exactly equal."""
    _, cross = partition_rows(indexed, hours)
    grouped = {}
    for epoch in epochs:
        prices = epoch["multipliers"]
        if len(prices) != len(indexed.rows) or any(not math.isfinite(p) for p in prices):
            raise ValueError("invalid multiplier vector")
        if any(not indexed.rows[i].equality and prices[i] < -1e-6 for i in cross):
            raise ValueError("negative inequality multiplier")
        key = tuple(prices[i] if indexed.rows[i].equality else max(0., prices[i]) for i in cross)
        if key not in grouped:
            grouped[key] = copy.deepcopy(epoch)
            grouped[key]["source_epochs"] = [epoch["outer"]]
            grouped[key]["price_key"] = key
            continue
        current = grouped[key]
        current["outer"] = epoch["outer"]
        current["source_clock"] = max(current["source_clock"], epoch["source_clock"])
        current["source_epochs"].append(epoch["outer"])
        for block, other in zip(current["blocks"], epoch["blocks"]):
            if block["times"] != other["times"] or abs(block["constant"]-other["constant"]) > 1e-5:
                raise ValueError("same-price snapshot partition mismatch")
            block["cuts"].extend(copy.deepcopy(other["cuts"]))
            block["lower"] = max(block["lower"], other["lower"])
            if other["best"] is not None and (block["best"] is None or other["best"]["cost"] < block["best"]["cost"]):
                block["best"] = copy.deepcopy(other["best"])
    return list(grouped.values())


def certify_views(data, scenarios, definition, views, deadline, publish):
    """Only ObjBound is observed. No primal point or new cut is produced."""
    indexed, cache = IndexedUC(data, scenarios), {}
    results, errors, records = {}, [], []
    for label, view in views.items():
        initial = view["initial_lower"]
        if not math.isfinite(initial):
            errors.append(f"{label}:missing_initial_continuous_bound")
            continue
        work = []
        try:
            grouped = coalesce_epochs(indexed, definition["hours"], view["epochs"])
        except (ValueError, KeyError, TypeError) as exc:
            grouped = []
            errors.append(f"{label}:{exc}")
        for epoch in grouped:
            try:
                models = list(verify_epoch(indexed, definition, view, epoch))
                work.append((epoch, models))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{label}/outer={epoch['outer']}:{exc}")
        work.sort(key=lambda item: (sum(b["lower"] for b in item[0]["blocks"]), item[0]["outer"]), reverse=True)
        result = results[label] = dict(lower=initial, coordination_upper=view["coordination_upper"],
            epochs=[], certified_models=0, total_models=sum(len(models) for _, models in work),
            complete=False, source_clock=None)
        for epoch, models in work:
            bounds = [saved["lower"] for _, _, saved in models]
            epoch_result = dict(outer=epoch["outer"], source_epochs=epoch["source_epochs"],
                                lower=sum(bounds), source_clock=epoch["source_clock"], blocks=[])
            result["epochs"].append(epoch_result)
            for j, (block, cuts, saved) in enumerate(models):
                row = dict(block=j, lower=saved["lower"], bound_source="continuous_snapshot",
                           cuts=len(cuts), status=None, seconds=0., cached=False)
                if time.perf_counter() < deadline:
                    key = epoch["price_key"], j, len(cuts)
                    phase = time.perf_counter()
                    try:
                        certificate = cache.get(key)
                        row["cached"] = certificate is not None
                        if certificate is None:
                            certificate = solve_master(block, cuts, deadline, integral=True,
                                                       bound_only=True, max_seconds=2.)
                        if "point" in certificate:
                            raise ValueError("primal channel exposed by offline certificate")
                        row.update(status=certificate["status"], seconds=time.perf_counter()-phase)
                        if time.perf_counter() >= deadline:
                            row["reason"] = "late_certificate_discarded"
                        else:
                            bound = certificate.get("lower_bound")
                            if bound is not None and math.isfinite(bound):
                                cache[key] = certificate
                                row["integer_lower"] = bound-1e-5
                                if bound-1e-5 > row["lower"]:
                                    row.update(lower=bound-1e-5, bound_source="offline_integer_master")
                                result["certified_models"] += 1
                            else:
                                row["reason"] = "missing_finite_ObjBound"
                            if certificate["status"] == 3 and saved["best"] is not None:
                                errors.append(f"{label}/outer={epoch['outer']}/block={j}:infeasible_master_with_witness")
                    except Exception as exc:
                        row["reason"] = type(exc).__name__+":"+str(exc)
                        errors.append(f"{label}/outer={epoch['outer']}/block={j}:{row['reason']}")
                else:
                    row["reason"] = "audit_budget_exhausted"
                if saved["best"] is not None and row["lower"] > saved["best"]["cost"]+1e-5:
                    errors.append(f"{label}/outer={epoch['outer']}/block={j}:integer_lower_exceeds_witness")
                bounds[j] = row["lower"]
                epoch_result["blocks"].append(row)
                epoch_result["lower"] = sum(bounds)
                if epoch_result["lower"] > result["lower"]:
                    result.update(lower=epoch_result["lower"], source_clock=epoch["source_clock"])
                records.append(dict(checkpoint=label, outer=epoch["outer"], block=j,
                    source_clock=epoch["source_clock"], audit_clock=time.perf_counter(),
                    reliable_lower=result["lower"], epoch_lower=epoch_result["lower"],
                    **{k:v for k,v in row.items() if k != "block"}))
                if time.perf_counter() < deadline:
                    publish(dict(checkpoints=results, errors=list(errors), records=records))
        result["complete"] = result["certified_models"] == result["total_models"]
    return dict(checkpoints=results, errors=errors, records=records)
