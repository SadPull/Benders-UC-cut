"""Time-block Benders/SA with Lagrangian cutting-plane coordination."""
from __future__ import annotations

from dataclasses import replace
import math
import time

import numpy as np

from .candidates import repair_commitment, commitment_key, commitment_hash
from .advisor_feedback import summarize, call_due
from .subagent import cancelled, publish
from .context import LLMContextInput, CandidateEvaluation, build_llm_context
from .llm import ask_llm_for_candidates
from .rule_advisor import rule_candidates
from .paper_model import IndexedUC, CompactQubo, key
from .block_master_model import continuous_relaxation, priced_blocks, partition_rows, certify_master, solve_master, point_record, block_dual_floor
from .coordination import converged, cutting_plane, coordinate, ABSOLUTE_TOLERANCE
from .quantum import solve_qubo
from .subproblem import solve_scenario


def project(candidate, block):
    return {k: int(getattr(candidate, k[0])[k[1], k[2]]) for k in block.bkeys}


def solve_paper(data, scenarios, config):
    from .solver import BendersResult, IterationRecord, ScenarioSolution, _construct_feasible_candidate, _fixed_cost

    started = time.perf_counter()
    deadline = started + (config.time_limit_seconds if config.time_limit_seconds is not None else float("inf"))
    sa_calls = llm_calls = stalled = batches = 0
    native_enabled = config.native_fallback_patience is not None
    variant = config.gap_experiment_variant
    pure_sa = variant is not None and variant.startswith("sa_")
    inexact = variant is not None and variant != "sa_strict"
    retained = 4 if inexact else 16
    defer_recovery = config.defer_full_recovery or inexact
    integer_certifications = 0
    native_calls, best_sources = 0, set()
    uc = IndexedUC(data, scenarios)
    local_rows, cross = partition_rows(uc, config.block_hours)
    diagnostics = {k: [] for k in ("evaluations", "proposals", "sampling", "iterations", "llm_calls", "outer", "blocks", "local_evaluations")}
    timings = diagnostics["timings"] = {k: 0. for k in ("qubo", "annealing", "relaxation", "local_dispatch", "repair", "subproblems", "llm", "context", "heuristic", "validation", "certification", "coordination")}
    incumbent, best, best_solutions = float("inf"), None, []
    evaluated, history, context_evaluations, best_results = set(), [], [], []
    optimality_count = feasibility_count = total_local = 0
    reliable_lower, coordination_upper = -float("inf"), float("inf")
    previous_points = [[] for _ in local_rows]
    previous_sources = [{} for _ in local_rows]
    diagnostics["native_candidates"] = []
    timings["native_master"] = 0.
    planes = [[] for _ in local_rows]
    previous_feedback = None
    status, failure_stage, failure_message = "MAX_ITERATIONS", None, ""
    multipliers = [0.] * len(uc.rows)

    if config.llm.schedule not in {"fixed", "stagnation", "stagnation_burst"} or (config.llm.max_calls is not None and config.llm.max_calls < 0):
        raise ValueError("invalid LLM schedule or call limit")

    def emit(kind, **payload):
        if config.event_sink is not None:
            config.event_sink({"kind": kind, "clock": time.perf_counter(), "elapsed_seconds": time.perf_counter()-started,
                               "sa_calls": sa_calls, **payload})

    def resource_status():
        if cancelled(config.llm):
            return "CANCELLED"
        if time.perf_counter() >= deadline:
            return "TIME_LIMIT"
        if config.max_candidate_evaluations is not None and len(evaluated) >= config.max_candidate_evaluations:
            return "EVALUATION_LIMIT"
        return None

    def available():
        return resource_status() is None

    def evaluate(candidate, outer, sources=None):
        nonlocal incumbent, best, best_solutions, best_results, best_sources
        if candidate is None or not available():
            return False
        signature = commitment_key(candidate.u, candidate.v, candidate.w)
        if signature in evaluated:
            return False
        before = incumbent
        phase = time.perf_counter()
        results = []
        for scenario in scenarios:
            if time.perf_counter() >= deadline:
                timings["subproblems"] += time.perf_counter()-phase
                emit("evaluation_incomplete", source=candidate.source, reason="deadline")
                return False
            emit("dispatch_start", scope="full", scenario=scenario.name)
            results.append(solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w,
                                         deadline=deadline, threads=1))
        elapsed = time.perf_counter()-phase
        timings["subproblems"] += elapsed
        # An interrupted solve is not a complete statistical evaluation.
        if time.perf_counter() >= deadline or any(r.status.startswith("TIME_LIMIT") or "TIME_LIMIT" in r.status for r in results):
            emit("evaluation_incomplete", source=candidate.source, reason="dispatch_timeout")
            return False
        evaluated.add(signature)
        feasible = all(r.feasible for r in results)
        fixed = _fixed_cost(data, candidate.u, candidate.v, candidate.w)
        fuel = sum(s.probability*r.objective for s, r in zip(scenarios, results)) if feasible else float("inf")
        objective = fixed+fuel
        solutions = [ScenarioSolution(s.name, s.probability, r.objective, r.p, r.r) for s, r in zip(scenarios, results)] if feasible else []
        if feasible:
            phase = time.perf_counter()
            from .validation import validate_schedule
            from types import SimpleNamespace
            validate_schedule(data, scenarios, SimpleNamespace(objective=objective, u=candidate.u,
                              v=candidate.v, w=candidate.w, scenarios=solutions))
            timings["validation"] += time.perf_counter()-phase
        if time.perf_counter() >= deadline:
            emit("evaluation_incomplete", source=candidate.source, reason="validation_deadline")
            return False
        if objective < incumbent:
            incumbent, best = objective, candidate
            best_solutions, best_results = solutions, results
            best_sources = set(sources or [candidate.source])
        context_evaluations.append(CandidateEvaluation(candidate.source, objective, fixed, fuel,
            {s.name:r.objective for s,r in zip(scenarios, results)}, feasible, 0, 0, None))
        diagnostics["evaluations"].append({"iteration": outer, "evaluation_count": len(evaluated),
            "elapsed_seconds": time.perf_counter()-started, "cache_hit": False, "source": candidate.source,
            "feasible": feasible, "objective": objective, "incumbent_before": before, "incumbent": incumbent,
            "improved": incumbent < before, "fixed_cost": fixed, "expected_fuel_cost": fuel,
            "commitment_hash": commitment_hash(candidate), "commitment_id": str(signature), "u": candidate.u, "v": candidate.v, "w": candidate.w,
            "subproblem_seconds": elapsed, "sa_calls": sa_calls})
        if native_enabled or variant:
            diagnostics["evaluations"][-1]["candidate_origins"] = sorted(sources or [candidate.source])
        emit("evaluation", **{k:v for k,v in diagnostics["evaluations"][-1].items() if k != "elapsed_seconds"},
             scenarios=solutions)
        return True

    def recover(points, outer, source, sources=None):
        if best is None or not available():
            return None
        u = best.u.copy()
        for point in points:
            for (kind, i, t, _), value in point.items():
                if kind == "u":
                    u[i, t] = int(value)
        phase = time.perf_counter()
        candidate = repair_commitment(data, scenarios, u)
        timings["repair"] += time.perf_counter()-phase
        if candidate is not None:
            if native_enabled or variant:
                sources = set(sources or [source])
                covered = {k for point in points for k in point if k[0] == "u"}
                if len(covered) < data.n_units*data.horizon:
                    sources.update(best_sources)
                if "native_master" in sources:
                    source = "mixed_native" if "sa" in sources else "native_master"
                elif variant:
                    changed = bool(np.any(candidate.u != u))
                    if changed:
                        sources.add("deterministic_repair")
                    source = ("sa_repaired" if changed else "sa") if "sa" in sources else "initialization"
            candidate.source = source
        evaluate(candidate, outer, sources)
        return candidate

    emit("started", deadline_clock=deadline, block_count=len(local_rows), block_hours=config.block_hours)
    try:
        if available():
            phase = time.perf_counter()
            relaxation = continuous_relaxation(uc, deadline)
            timings["relaxation"] += time.perf_counter()-phase
            multipliers = relaxation["multipliers"]
            initial_local_prices = list(multipliers)
            if variant:
                emit("initial_dual_prices", multipliers=initial_local_prices)
            # The relaxed optimum bounds the integer primal even if no block finishes.
            reliable_lower = relaxation["lower_bound"] - ABSOLUTE_TOLERANCE
            diagnostics["initialization"] = {k: relaxation[k] for k in ("objective", "lower_bound", "seconds")}
            emit("bound_update", outer=0, block=None, inner=0, reliable_lower=reliable_lower,
                 bound_source="continuous_relaxation", coordination_upper=None)
            u = np.array([[round(relaxation["point"][key("u", i, t)]) for t in range(data.horizon)] for i in range(data.n_units)])
            rounded = repair_commitment(data, scenarios, u)
            if rounded is not None:
                rounded.source = "relaxation_rounding"
            evaluate(rounded, 0)
            evaluate(_construct_feasible_candidate(data, scenarios), 0)
        if best is None and available():
            status, failure_stage = "INITIALIZATION_FAILED", "initialization"
    except Exception as exc:
        status, failure_stage, failure_message = "INITIALIZATION_FAILED", "initialization", str(exc)

    for outer in range(1, config.max_iterations+1):
        if best is None or failure_stage or not available():
            break
        if config.advisor_only and config.llm.max_calls is not None and llm_calls >= config.llm.max_calls:
            status = "LLM_CALL_LIMIT"
            break
        blocks, _ = priced_blocks(uc, config.block_hours, multipliers)
        current_bounds = [block.objective_bounds(block.costs)[0]+block.constant for block in blocks]
        if inexact:
            current_bounds = [max(bound, block_dual_floor(block, initial_local_prices))
                              for block, bound in zip(blocks, current_bounds)]
        if variant:
            emit("multiplier_epoch", outer=outer, multipliers=list(multipliers),
                 blocks=[dict(block=j, times=list(block.times), constant=block.constant,
                              lower=current_bounds[j]) for j, block in enumerate(blocks)])
        pools, lower_bounds, upper_bounds = [], [], []
        before_count, before_opt, before_feas = len(evaluated), optimality_count, feasibility_count
        round_advice = []
        outer_record = dict(outer=outer, multipliers=list(multipliers), completed_blocks=0,
                            dual_interval=None, coordination_upper=None, converged=False)
        diagnostics["outer"].append(outer_record)
        for b, block in enumerate(blocks):
            if not available():
                break
            # All price-dependent cuts and costs are rebuilt for every multiplier.
            cuts, cache, pool = [], {}, []
            stalled_inner = 0
            analytic_lo, analytic_hi = block.objective_bounds(block.costs)
            lb, ub = current_bounds[b], float("inf")
            block_record = dict(outer=outer, block=b, times=list(block.times), constant=block.constant,
                                lower=lb, upper=ub, converged=False, inner_iterations=0,
                                certification_seconds=0.)
            diagnostics["blocks"].append(block_record)
            emit("block_start", **block_record)

            def local_evaluate(x, inner, origin, sources=None):
                nonlocal ub, total_local, optimality_count, feasibility_count
                x = {k: x[k] for k in block.bkeys}
                signature = tuple(x[k] for k in block.bkeys)
                if signature in cache or not available() or not block.discrete_checks(x)["feasible"]:
                    return
                phase = time.perf_counter()
                emit("dispatch_start", scope="block", outer=outer, block=b, inner=inner, origin=origin)
                result = block.dispatch(x, block.costs, deadline)
                timings["local_dispatch"] += time.perf_counter()-phase
                if result is None or not available():
                    return
                result["sources"] = set(sources or [origin])
                total_local += 1
                cache[signature] = result
                cuts.append(result["cut"])
                if result["feasible"]:
                    optimality_count += 1
                    # dispatch cost excludes the Lagrangian constant.
                    ub = min(ub, result["cost"] + block.constant)
                    pool.append(result)
                else:
                    feasibility_count += 1
                if variant:
                    cut = result["cut"]
                    emit("local_cut", outer=outer, block=b, inner=inner, origin=origin,
                         cut=dict(a=point_record(cut.a), c=cut.c, feasible=cut.feasible),
                         point=point_record(result["point"]), sources=sorted(result["sources"]),
                         feasible=result["feasible"],
                         cost=result["cost"]+block.constant if result["feasible"] else None)
                diagnostics["local_evaluations"].append(dict(outer=outer, block=b, inner=inner,
                    origin=origin, feasible=result["feasible"],
                    cost=result["cost"]+block.constant if result["feasible"] else None,
                    local_evaluation_count=total_local, elapsed_seconds=time.perf_counter()-started))
                return result

            local_evaluate(project(best, block), 0, "incumbent_projection", best_sources)
            for point in previous_points[b]:
                local_evaluate(point, 0, "previous_candidate",
                               previous_sources[b].get(tuple(point[k] for k in block.bkeys)))
            inner_limit = min(8, config.max_inner_iterations) if inexact else config.max_inner_iterations
            for inner in range(1, inner_limit+1):
                if not available():
                    break
                if config.advisor_only and config.llm.max_calls is not None and llm_calls >= config.llm.max_calls:
                    status = "LLM_CALL_LIMIT"
                    break
                batches += 1
                sa_before = incumbent
                local_before = ub
                phase = time.perf_counter()
                try:
                    lp_value = block.relaxation(block.costs, cuts, deadline)
                except Exception as exc:
                    lp_value = None
                    failure_message = str(exc)
                timings["relaxation"] += time.perf_counter()-phase
                if lp_value is None:
                    if variant and not available():
                        emit("computation_incomplete", outer=outer, block=b, inner=inner,
                             reason="continuous_relaxation_deadline")
                        break
                    status, failure_stage = "RELAXATION_FAILED", f"outer={outer}/block={b}/inner={inner}/relaxation"
                    break
                lb = max(lb, lp_value + block.constant - ABSOLUTE_TOLERANCE)
                if not available():
                    break
                if config.quantum_candidates:
                    phase = time.perf_counter()
                    # Remove the constant for encoding, restore only in reported bounds.
                    master = CompactQubo(block, block.costs, cuts, lb-block.constant,
                        min(ub-block.constant, analytic_hi), config)
                    build_seconds = time.perf_counter()-phase
                    timings["qubo"] += build_seconds
                    if not available():
                        break
                    qconfig = config.quantum
                    if config.sa_seed_per_call:
                        qconfig = replace(qconfig, kaiwu_seed=qconfig.kaiwu_seed+sa_calls)
                    phase = time.perf_counter()
                    sa_calls += 1
                    emit("sa_start", iteration=outer, outer=outer, block=b, inner=inner, bits=len(master.names), seed=qconfig.kaiwu_seed)
                    try:
                        binaries = solve_qubo(master.matrix, qconfig, energy_unit=master.energy_unit)
                    except Exception as exc:
                        status, failure_stage = "SAMPLING_FAILED", f"outer={outer}/block={b}/inner={inner}/sampling"
                        failure_message = str(exc)
                        emit("sa_end", status="failed", error=type(exc).__name__)
                        break
                    sampling_seconds = time.perf_counter()-phase
                    timings["annealing"] += sampling_seconds
                    emit("sa_end", status="completed", outer=outer, block=b, inner=inner)
                    decoded = [master.decode(bits)[0] for bits in binaries]
                    row = dict(iteration=outer, outer=outer, block=b, inner=inner, bits=len(master.names),
                        commitment_bits=len(master.vars), objective_bits=len(master.weights),
                        decision_bits=len(master.vars)+len(master.weights),
                        auxiliary_bits=len(master.names)-len(master.vars)-len(master.weights),
                        raw_candidates=len(decoded), raw_discrete_feasible=sum(block.discrete_checks(x)["feasible"] for x in decoded),
                        unique_uvw=len({tuple(x[k] for k in block.bkeys) for x in decoded}),
                        lower=master.lower, upper=master.upper, lp_lower=lp_value,
                        build_seconds=build_seconds, sampling_seconds=sampling_seconds, cuts=len(cuts),
                        elapsed_seconds=time.perf_counter()-started)
                    diagnostics["sampling"].append(row)
                    emit("sampling", **{k:v for k,v in row.items() if k != "elapsed_seconds"})
                    for x in decoded:
                        checks = block.discrete_checks(x)
                        proposal = dict(outer=outer, block=b, inner=inner, origin="sa",
                            proposal_id=f"sa:{sa_calls}:{len(diagnostics['proposals'])}",
                            raw_uvw=[int(x[k]) for k in block.bkeys], raw_discrete_feasible=checks["feasible"],
                            duplicate=tuple(x[k] for k in block.bkeys) in cache, accepted=checks["feasible"])
                        local_evaluate(x, inner, "sa")
                        result = cache.get(tuple(x[k] for k in block.bkeys))
                        candidate = recover([result["point"]], outer, "sa", result["sources"]) if result and result["feasible"] and not defer_recovery else None
                        proposal["full_recovery_deferred"] = defer_recovery
                        proposal.update(repaired_discrete_feasible=None if defer_recovery else candidate is not None,
                            commitment_id=commitment_hash(candidate) if candidate else None,
                            hamming_u=sum(int(candidate.u[i,t]) != x[key("u",i,t)]
                                          for i in range(data.n_units) for t in block.times) if candidate else None)
                        diagnostics["proposals"].append(proposal)
                        emit("proposal", **proposal)
                # Full-horizon validation is independent of local priced evaluation.
                base_for_llm = []
                for entry in ([] if defer_recovery else sorted(pool, key=lambda r: r["cost"])[:16]):
                    candidate = recover([entry["point"]], outer, "pool" if config.advisor_only else "sa", entry["sources"])
                    if candidate is not None:
                        base_for_llm.append(candidate)
                if (config.llm.enabled or config.rule_advisor) and available():
                    due, stalled = call_due(config.llm.schedule, sa_before, incumbent, stalled, llm_calls)
                    emit("sa_round_validated", iteration=outer, outer=outer, block=b, inner=inner,
                         batch=batches, incumbent_before=sa_before, incumbent=incumbent, stalled=stalled, llm_due=due)
                    burst_size = 3 if config.llm.schedule == "stagnation_burst" else 1
                    batch_advice = []
                    for batch_index in range(1, burst_size+1 if config.advisor_only or due else 1):
                        if not available() or (config.llm.max_calls is not None and llm_calls >= config.llm.max_calls):
                            break
                        phase = time.perf_counter()
                        context = build_llm_context(data, scenarios, LLMContextInput(
                            incumbent=best, incumbent_results=best_results, base_evaluations=context_evaluations[-16:],
                            candidate_evaluations=context_evaluations, history=history))
                        context["block_cut_count"] = len(cuts)
                        if config.llm.enhanced_feedback:
                            context["previous_call_feedback"] = previous_feedback
                        baseline_u = best.u.copy()
                        baseline_row = next(r for r in reversed(diagnostics["evaluations"]) if r["commitment_hash"] == commitment_hash(best))
                        baseline_cost = dict(total=baseline_row["objective"], fixed=baseline_row["fixed_cost"], expected_fuel=baseline_row["expected_fuel_cost"])
                        prior_ids = {r["commitment_hash"] for r in diagnostics["evaluations"]}
                        timings["context"] += time.perf_counter()-phase
                        phase = time.perf_counter()
                        advisor_kind = "rule" if config.rule_advisor else "llm"
                        llm_calls += 1
                        call_index = llm_calls
                        emit(f"{advisor_kind}_start", iteration=outer, outer=outer, block=b, inner=inner, call_index=call_index, batch_index=batch_index)
                        timeout = config.llm.timeout_seconds
                        if math.isfinite(deadline):
                            remaining = max(.001, deadline-time.perf_counter())
                            timeout = remaining if timeout is None else min(timeout, remaining)
                        if config.rule_advisor:
                            advice = rule_candidates(data, scenarios, best, context, prior_ids, call_index, config.llm.max_candidates, deadline)
                        else:
                            advice = ask_llm_for_candidates(data, scenarios, call_index, base_for_llm, best,
                                replace(config.llm, timeout_seconds=timeout, max_retries=1, collect_diagnostics=True), context=context)
                        batch_advice.append(advice)
                        round_advice.append(advice)
                        timings["heuristic" if config.rule_advisor else "llm"] += time.perf_counter()-phase-advice.diagnostics.get("repair_seconds", 0.)
                        timings["repair"] += advice.diagnostics.get("repair_seconds", 0.)
                        diagnostics["llm_calls"].append(advice.diagnostics)
                        emit(f"{advisor_kind}_end", diagnostics=advice.diagnostics)
                        for candidate in advice.candidates:
                            if not available():
                                break
                            x = project(candidate, block)
                            local_evaluate(x, inner, advisor_kind)
                            local_result = cache.get(tuple(x[k] for k in block.bkeys))
                            for proposal in advice.diagnostics.get("proposals", []):
                                if proposal.get("commitment_id") == commitment_hash(candidate):
                                    proposal["local_dispatch_completed"] = local_result is not None
                                    proposal["local_feasible"] = local_result["feasible"] if local_result else None
                            evaluate(candidate, outer)
                        previous_feedback = summarize(advice, baseline_u, baseline_cost, diagnostics["evaluations"], prior_ids)
                        advice.diagnostics["feedback"] = previous_feedback
                        emit(f"{advisor_kind}_feedback", feedback=previous_feedback)
                        if not config.rule_advisor and config.llm.api_format == "codex_subagent":
                            from pathlib import Path
                            call_dir = Path(config.llm.exchange_dir) / "calls" / f"{call_index:03d}"
                            publish(call_dir / "feedback.json", previous_feedback)
                    if batch_advice:
                        stalled = 0
                if not available():
                    break
                if config.advisor_only and config.llm.max_calls is not None and llm_calls >= config.llm.max_calls:
                    status = "LLM_CALL_LIMIT"
                    break
                native_record = None
                if native_enabled:
                    if pure_sa:
                        raise AssertionError("native integer candidate called in pure-SA search")
                    stalled_inner = 0 if ub < local_before-ABSOLUTE_TOLERANCE else stalled_inner+1
                    if not config.quantum_candidates or stalled_inner >= config.native_fallback_patience:
                        native_calls += 1
                        native_record = dict(outer=outer, block=b, inner=inner, call=native_calls,
                            reason="stagnation" if config.quantum_candidates else "native_only",
                            lower_before=lb, upper_before=ub, accepted=False, duplicate=False)
                        phase = time.perf_counter()
                        emit("native_start", **native_record)
                        try:
                            oracle = solve_master(block, cuts, deadline, bound_only=False)
                            native_record["seconds"] = time.perf_counter()-phase
                            timings["native_master"] += native_record["seconds"]
                            native_record.update(status=oracle["status"], oracle_seconds=oracle["seconds"],
                                objective=oracle.get("objective"), mip_gap=oracle.get("mip_gap"))
                            point = oracle.get("point")
                            if not available():
                                native_record["rejection"] = "deadline"
                            elif point is None:
                                native_record["rejection"] = "missing_point"
                            else:
                                signature = tuple(point[k] for k in block.bkeys)
                                native_record["raw_uvw"] = list(signature)
                                native_record["duplicate"] = signature in cache
                                entry = local_evaluate(point, inner, "native_master")
                                native_record["accepted"] = entry is not None
                                if entry and entry["feasible"] and not defer_recovery:
                                    recover([entry["point"]], outer, "native_master", entry["sources"])
                        except Exception as exc:
                            native_record["rejection"] = type(exc).__name__
                            native_record["error"] = str(exc)
                        if "seconds" not in native_record:
                            native_record["seconds"] = time.perf_counter()-phase
                            timings["native_master"] += native_record["seconds"]
                        stalled_inner = 0
                        diagnostics["native_candidates"].append(native_record)
                if not available():
                    if native_record is not None:
                        native_record.update(lower_after=lb, upper_after=ub)
                        emit("native_candidate", **native_record)
                    break
                try:
                    if pure_sa:
                        certificate = dict(lower_bound=lb, seconds=0., status="LP_ONLY")
                    else:
                        integer_certifications += 1
                        certificate = certify_master(block, cuts, deadline)
                        if variant:
                            emit("integer_certification", outer=outer, block=b, inner=inner,
                                 seconds=certificate["seconds"])
                    timings["certification"] += certificate["seconds"]
                    block_record["certification_seconds"] += certificate["seconds"]
                    bound = certificate["lower_bound"]
                    if bound is None or not math.isfinite(bound):
                        raise ValueError("integer master returned no finite ObjBound")
                    lb = max(lb, bound)
                    if lb > ub + ABSOLUTE_TOLERANCE:
                        raise ValueError(f"inconsistent certified bounds: {lb} > {ub}")
                except Exception as exc:
                    status, failure_stage = "CERTIFICATION_FAILED", f"outer={outer}/block={b}/inner={inner}/certification"
                    failure_message = str(exc)
                    if native_record is not None:
                        native_record.update(lower_after=lb, upper_after=ub, certification_error=str(exc))
                        emit("native_candidate", **native_record)
                    break
                done = converged(lb, ub, config.convergence_tolerance)
                current_bounds[b] = lb
                reliable_lower = max(reliable_lower, sum(current_bounds))
                if native_record is not None:
                    native_record.update(lower_after=lb, upper_after=ub)
                    emit("native_candidate", **native_record)
                emit("bound_update", outer=outer, block=b, inner=inner, reliable_lower=reliable_lower,
                     bound_source="same_multiplier_block_certificates", block_bounds=list(current_bounds),
                     lower=lb, upper=ub, coordination_upper=None)
                row = dict(outer=outer, block=b, inner=inner, lower=lb, upper=ub, converged=done,
                           cuts=len(cuts), certificate=certificate)
                diagnostics["iterations"].append(row)
                block_record.update(lower=lb, upper=ub, converged=done, inner_iterations=inner)
                emit("inner_iteration", **row)
                if variant:
                    emit("block_bound", outer=outer, block=b, inner=inner, lower=lb,
                         upper=ub, converged=done)
                if done:
                    break
            block_record.update(lower=lb, upper=ub)
            pool.sort(key=lambda r: r["cost"])
            if defer_recovery and not inexact:
                emit("deferred_recovery_start", outer=outer, block=b, candidates=min(16, len(pool)))
                for entry in pool[:16]:
                    if not available():
                        break
                    recover([entry["point"]], outer, "sa", entry["sources"])
                emit("deferred_recovery_end", outer=outer, block=b)
            previous_points[b] = [{k:r["point"][k] for k in block.bkeys} for r in pool[:retained]]
            previous_sources[b] = {tuple(r["point"][k] for k in block.bkeys): r["sources"] for r in pool[:retained]}
            if variant:
                emit("block_bound", outer=outer, block=b, inner=block_record["inner_iterations"],
                     lower=lb, upper=ub, converged=block_record["converged"])
            if failure_stage or not available() or (not block_record["converged"] and not inexact):
                if status == "MAX_ITERATIONS" and available():
                    status = "INNER_NOT_CONVERGED"
                break
            if not pool:
                status, failure_stage = "LOCAL_POOL_EMPTY", f"outer={outer}/block={b}/local_pool"
                break
            pools.append(pool[:retained])
            lower_bounds.append(lb)
            upper_bounds.append(ub)
            outer_record["completed_blocks"] = len(pools)
        if len(pools) != len(blocks) or not available():
            break
        reliable_lower = max(reliable_lower, sum(lower_bounds))
        outer_record["dual_interval"] = [sum(lower_bounds), sum(upper_bounds)]
        for rank in range(0 if inexact else min(16, max(map(len, pools)))):
            entries = [pool[min(rank, len(pool)-1)] for pool in pools]
            recover([entry["point"] for entry in entries], outer, "stitched",
                    set().union(*(entry["sources"] for entry in entries)))
        if not available():
            break
        try:
            for b, block in enumerate(blocks):
                planes[b].append(cutting_plane(block, cross, pools[b][0]["point"]))
            coordination = coordinate(uc, cross, planes, incumbent, deadline)
            timings["coordination"] += coordination["seconds"]
            coordination_upper = coordination["upper"]
            if coordination_upper < reliable_lower - ABSOLUTE_TOLERANCE:
                raise ValueError("coordination upper bound is below the certified lower bound")
        except Exception as exc:
            status, failure_stage, failure_message = "COORDINATION_FAILED", f"outer={outer}/coordination", str(exc)
            break
        if inexact:
            # Coordination uses feasible upper planes even when inner solves are inexact.
            emit("approximate_outer", outer=outer, lower=reliable_lower,
                 upper=coordination_upper, all_inner_converged=all(
                     r["converged"] for r in diagnostics["blocks"] if r["outer"] == outer))
            for rank in range(min(2, max(map(len, pools)))):
                if not available():
                    break
                entries = [pool[min(rank, len(pool)-1)] for pool in pools]
                recover([entry["point"] for entry in entries], outer, "stitched",
                        set().union(*(entry["sources"] for entry in entries)))
            coordination_upper = min(coordination_upper, incumbent)
        done = converged(reliable_lower, coordination_upper, config.convergence_tolerance, reference=reliable_lower)
        outer_record.update(reliable_lower=reliable_lower, coordination_upper=coordination_upper,
                            converged=done, elapsed_seconds=time.perf_counter()-started)
        history.append(IterationRecord(outer, sum(len(p) for p in pools), 0, len(evaluated)-before_count,
            len(evaluated)-before_count, 0, optimality_count-before_opt, feasibility_count-before_feas, incumbent,
            llm_candidates=sum(a.raw_count for a in round_advice), llm_valid_candidates=sum(a.valid_count for a in round_advice),
            llm_failed=any(a.failed for a in round_advice)))
        emit("iteration", iteration=outer, incumbent=incumbent, reliable_lower=reliable_lower,
             coordination_upper=coordination_upper, dual_interval=outer_record["dual_interval"], timings=dict(timings))
        emit("bound_update", outer=outer, block=None, inner=None, reliable_lower=reliable_lower,
             bound_source="completed_outer", coordination_upper=coordination_upper)
        if done:
            status = "DUAL_CONVERGED"
            break
        multipliers = coordination["multipliers"]
        if variant:
            emit("multiplier_update", outer=outer, next_outer=outer+1,
                 max_abs_price=max(map(abs, multipliers), default=0.))
    status = resource_status() or status
    runtime = time.perf_counter()-started
    diagnostics.update(evaluation_count=len(evaluated), local_evaluation_count=total_local,
        elapsed_seconds=runtime, time_limit_overshoot=max(0., time.perf_counter()-deadline),
        reliable_lower=reliable_lower, coordination_upper=coordination_upper,
        primal_gap=(incumbent-reliable_lower)/max(1., abs(incumbent)) if math.isfinite(incumbent) and math.isfinite(reliable_lower) else None,
        block_hours=config.block_hours, block_count=len(local_rows), coupling_count=len(cross),
        termination_reason=status, failure_stage=failure_stage, failure_message=failure_message,
        sa_calls=sa_calls, native_calls=native_calls, llm_call_count=0 if config.rule_advisor else llm_calls,
        rule_call_count=llm_calls if config.rule_advisor else 0)
    if variant:
        diagnostics.update(gap_experiment_variant=variant, integer_certifications=integer_certifications,
                           online_integer_master_calls=native_calls+integer_certifications)
    timings["other"] = max(0., runtime-sum(timings.values()))
    empty = np.empty((0, 0), dtype=int)
    return BendersResult(status, incumbent, best.u if best else empty, best.v if best else empty,
        best.w if best else empty, best_solutions, history, optimality_count, feasibility_count,
        runtime, message=failure_message, diagnostics=diagnostics)
