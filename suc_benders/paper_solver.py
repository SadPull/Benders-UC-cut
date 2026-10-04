"""Dynamic whole-horizon SA search with validated LLM candidates."""
from __future__ import annotations

from dataclasses import replace
import math
import time
from itertools import count

import numpy as np

from .candidates import repair_commitment, commitment_key, make_candidate, commitment_hash
from .advisor_feedback import summarize, call_due
from .subagent import cancelled, publish
from .context import LLMContextInput, CandidateEvaluation, build_llm_context
from .llm import ask_llm_for_candidates
from .rule_advisor import rule_candidates
from .paper_model import IndexedUC, Block, CompactQubo, key
from .quantum import solve_qubo
from .subproblem import solve_scenario


def solve_paper(data, scenarios, config):
    from .solver import BendersResult, IterationRecord, ScenarioSolution, _construct_feasible_candidate, _fixed_cost

    started = time.perf_counter()
    deadline = started + (config.time_limit_seconds if config.time_limit_seconds is not None else float("inf"))
    sa_calls = 0

    def emit(kind, **payload):
        if config.event_sink is not None:
            config.event_sink({"kind": kind, "elapsed_seconds": time.perf_counter()-started,
                               "sa_calls": sa_calls, **payload})

    emit("started", deadline_clock=deadline)
    uc = IndexedUC(data, scenarios)
    blocks = [Block(uc)]
    diagnostics = {"evaluations": [], "proposals": [], "sampling": [], "iterations": [],
                   "llm_calls": [], "outer": [], "local_evaluations": [],
                   "timings": {k: 0. for k in ("qubo", "annealing", "relaxation", "local_dispatch", "repair", "subproblems", "llm", "context", "heuristic", "validation")}}
    timings = diagnostics["timings"]
    incumbent, best, best_solutions = float("inf"), None, []
    evaluated = set()
    history = []
    optimality_count = feasibility_count = 0
    reliable_lower = -float("inf")
    total_local = 0
    persistent = [[] for _ in blocks]
    persistent_cache = [{} for _ in blocks]
    infeasible_cache = [set() for _ in blocks]
    previous_points = [[] for _ in blocks]
    best_results = []
    context_evaluations = []

    previous_feedback = None
    llm_calls = stalled = 0
    if config.llm.schedule not in {"fixed", "stagnation", "stagnation_burst"} or (config.llm.max_calls is not None and config.llm.max_calls < 0):
        raise ValueError("invalid LLM schedule or call limit")

    def available():
        return not (config.target_cost is not None and incumbent < config.target_cost) and not cancelled(config.llm) and time.perf_counter() < deadline and (config.max_candidate_evaluations is None or len(evaluated) < config.max_candidate_evaluations)

    def evaluate(candidate, outer):
        nonlocal incumbent, best, best_solutions, best_results
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
        if feasible and (config.event_sink is not None or config.target_cost is not None):
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
        context_evaluations.append(CandidateEvaluation(candidate.source, objective, fixed, fuel,
            {s.name:r.objective for s,r in zip(scenarios, results)}, feasible, 0, 0, None))
        diagnostics["evaluations"].append({"iteration": outer, "evaluation_count": len(evaluated),
            "elapsed_seconds": time.perf_counter()-started, "cache_hit": False, "source": candidate.source,
            "feasible": feasible, "objective": objective, "incumbent_before": before, "incumbent": incumbent,
            "improved": incumbent < before, "fixed_cost": fixed, "expected_fuel_cost": fuel,
            "commitment_hash": commitment_hash(candidate), "commitment_id": str(signature), "u": candidate.u, "v": candidate.v, "w": candidate.w,
            "subproblem_seconds": elapsed, "sa_calls": sa_calls})
        emit("evaluation", **{k:v for k,v in diagnostics["evaluations"][-1].items() if k != "elapsed_seconds"},
             scenarios=solutions)
        return True

    seed = _construct_feasible_candidate(data, scenarios)
    evaluate(seed, 0)
    status = "MAX_ITERATIONS"
    for outer in (count(1) if config.target_cost is not None or config.max_iterations is None else range(1, config.max_iterations+1)):
        if not available():
            break
        if config.advisor_only and config.llm.max_calls is not None and llm_calls >= config.llm.max_calls:
            status = "LLM_CALL_LIMIT"
            break
        sa_before = incumbent
        round_advice = []
        costs = uc.linear
        current = best if best is not None else seed
        if current is None:
            status = "NO_FEASIBLE_SOLUTION"
            break
        seed_point = {key(kind, i, t): int(getattr(current, kind)[i, t])
                      for kind in ("u", "v", "w") for i in range(data.n_units) for t in range(data.horizon)}
        pools, lower_bounds = [], []
        before_count, before_opt, before_feas = len(evaluated), optimality_count, feasibility_count
        for b, block in enumerate(blocks):
            if not available():
                break
            cuts = persistent[b]
            cache = persistent_cache[b]
            pool = [r for r in cache.values() if r["feasible"]]
            analytic_lo, analytic_hi = block.objective_bounds(costs)
            lb, ub = analytic_lo, analytic_hi
            if pool:
                ub = min(ub, min(r["cost"] for r in pool))

            def local_evaluate(x, inner, origin):
                nonlocal ub, total_local, optimality_count, feasibility_count
                signature = tuple(x[k] for k in block.bkeys)
                if signature in cache or signature in infeasible_cache[b] or not available() or not block.discrete_feasible(x):
                    return
                phase = time.perf_counter()
                emit("dispatch_start", scope="block", block=b, origin=origin)
                result = block.dispatch(x, costs, deadline)
                timings["local_dispatch"] += time.perf_counter()-phase
                if result is None:
                    return
                total_local += 1
                cache[signature] = result
                cuts.append(result["cut"])
                if result["feasible"]:
                    optimality_count += 1
                    ub = min(ub, result["cost"])
                    pool.append(result)
                else:
                    feasibility_count += 1
                    infeasible_cache[b].add(signature)
                diagnostics["local_evaluations"].append({"outer": outer, "block": b, "inner": inner,
                    "origin": origin,
                    "feasible": result["feasible"], "cost": result["cost"] if result["feasible"] else None,
                    "local_evaluation_count": total_local, "elapsed_seconds": time.perf_counter()-started})

            local_evaluate({k: seed_point[k] for k in block.bkeys}, 0, "incumbent_projection")
            for point in previous_points[b]:
                local_evaluate(point, 0, "previous_sa")
            for inner in range(1, (0 if config.advisor_only else 1)+1):
                if not available():
                    break
                lp_value = None
                lp_seconds = 0.
                phase = time.perf_counter()
                lp_value = block.relaxation(costs, cuts, deadline)
                lp_seconds = time.perf_counter()-phase
                timings["relaxation"] += lp_seconds
                if lp_value is not None:
                    lb = max(lb, lp_value-1e-5)
                if not available():
                    break
                phase = time.perf_counter()
                master = CompactQubo(block, costs, cuts,
                                     lb, ub, config)
                build_seconds = time.perf_counter()-phase
                timings["qubo"] += build_seconds
                if not available():
                    break
                qconfig = config.quantum
                if config.sa_seed_per_call:
                    qconfig = replace(qconfig, kaiwu_seed=qconfig.kaiwu_seed+sa_calls)
                phase = time.perf_counter()
                sa_calls += 1
                emit("sa_start", iteration=outer, bits=len(master.names), seed=qconfig.kaiwu_seed)
                try:
                    binaries = solve_qubo(master.matrix, qconfig, energy_unit=master.energy_unit)
                except Exception as exc:
                    emit("sa_end", status="failed", error=type(exc).__name__)
                    raise
                emit("sa_end", status="completed")
                sampling_seconds = time.perf_counter()-phase
                timings["annealing"] += sampling_seconds
                decoded = [master.decode(bits)[0] for bits in binaries]
                bit_count = len(master.names)
                diagnostics["sampling"].append({"iteration": outer, "block": b, "inner": inner,
                    "bits": bit_count, "decision_bits": len(master.vars)+len(master.weights),
                    "commitment_bits": len(master.vars), "objective_bits": len(master.weights),
                    "auxiliary_bits": bit_count-len(master.vars)-len(master.weights),
                    "density": float(np.count_nonzero(master.matrix)/master.matrix.size) if master.matrix.size else 0.,
                    "raw_candidates": len(decoded), "raw_discrete_feasible": sum(block.discrete_feasible(x) for x in decoded),
                    "unique_uvw": len({tuple(x[k] for k in block.bkeys) for x in decoded}),
                    "lower": master.lower, "upper": master.upper, "lp_lower": lp_value,
                    "lp_status": "OPTIMAL" if lp_value is not None else "UNAVAILABLE",
                    "lp_seconds": lp_seconds, "build_seconds": build_seconds, "sampling_seconds": sampling_seconds,
                    "cuts": len(cuts), "elapsed_seconds": time.perf_counter()-started})
                emit("sampling", **{k: v for k, v in diagnostics["sampling"][-1].items() if k != "elapsed_seconds"})
                for x in decoded:
                    if not available():
                        break
                    raw_u = np.array([[x[key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)])
                    phase = time.perf_counter()
                    repaired = repair_commitment(data, scenarios, raw_u)
                    timings["repair"] += time.perf_counter()-phase
                    signature = commitment_key(repaired.u, repaired.v, repaired.w) if repaired else None
                    local_signature = tuple(int(getattr(repaired, kind)[i, t])
                        for kind, i, t, _ in block.bkeys) if repaired else None
                    proposal = {"outer": outer, "origin": "sa", "raw_u": raw_u,
                        "proposal_id": f"sa:{sa_calls}:{len(diagnostics['proposals'])}",
                        "raw_discrete_feasible": block.discrete_feasible(x),
                        "repaired_discrete_feasible": repaired is not None,
                        "commitment_id": commitment_hash(repaired) if repaired else None,
                        "duplicate": bool(repaired and (signature in evaluated or local_signature in cache)),
                        "hamming_u": int(np.count_nonzero(repaired.u != raw_u)) if repaired else None,
                        "accepted": repaired is not None}
                    diagnostics["proposals"].append(proposal)
                    emit("proposal", **proposal)
                    if repaired is None:
                        continue
                    x = {key(kind, i, t): int(getattr(repaired, kind)[i, t]) for kind in ("u", "v", "w")
                         for i in range(data.n_units) for t in range(data.horizon)}
                    local_evaluate(x, inner, "sa")
            if (config.llm.enabled or config.rule_advisor) and available():
                # Complete SA evaluations first: LLM sees measured full-horizon feedback.
                base_for_llm = []
                for entry in sorted(pool, key=lambda r: r["cost"])[:config.quantum.candidate_pool_size]:
                    if not available():
                        break
                    u = np.array([[entry["point"][key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)])
                    candidate = make_candidate(data, scenarios, u, source="pool" if config.advisor_only else "sa")
                    evaluate(candidate, outer)
                    base_for_llm.append(candidate)
                due, stalled = call_due(config.llm.schedule, sa_before, incumbent, stalled, llm_calls)
                if not config.advisor_only:
                    emit("sa_round_validated", iteration=outer, incumbent_before=sa_before,
                         incumbent=incumbent, stalled=stalled, llm_due=due)
                burst_size = 3 if config.llm.schedule == "stagnation_burst" else 1
                for batch_index in range(1, burst_size+1 if config.advisor_only or due else 1):
                    if not available() or (config.llm.max_calls is not None and llm_calls >= config.llm.max_calls):
                        break
                    phase = time.perf_counter()
                    context = build_llm_context(data, scenarios, LLMContextInput(
                        incumbent=best, incumbent_results=best_results, base_evaluations=context_evaluations[-16:],
                        candidate_evaluations=context_evaluations,
                        history=history))
                    context["block_cut_count"] = len(cuts)
                    if config.llm.enhanced_feedback:
                        context["previous_call_feedback"] = previous_feedback
                    baseline_u = best.u.copy() if best is not None else None
                    baseline_row = next((r for r in reversed(diagnostics["evaluations"]) if best is not None and r["commitment_hash"] == commitment_hash(best)), None)
                    baseline_cost = dict(total=baseline_row["objective"], fixed=baseline_row["fixed_cost"], expected_fuel=baseline_row["expected_fuel_cost"]) if baseline_row else None
                    prior_ids = {r["commitment_hash"] for r in diagnostics["evaluations"]}
                    timings["context"] += time.perf_counter()-phase
                    phase = time.perf_counter()
                    advisor_kind = "rule" if config.rule_advisor else "llm"
                    emit(f"{advisor_kind}_start", iteration=outer, call_index=llm_calls+1, batch_index=batch_index)
                    llm_calls += 1
                    call_index = llm_calls if config.llm.schedule == "stagnation_burst" else outer
                    timeout = config.llm.timeout_seconds
                    if math.isfinite(deadline):
                        remaining = max(.001, deadline-time.perf_counter())
                        timeout = remaining if timeout is None else min(timeout, remaining)
                    if config.rule_advisor:
                        advice = rule_candidates(data, scenarios, best, context, prior_ids,
                                                 call_index, config.llm.max_candidates, deadline)
                    else:
                        advice = ask_llm_for_candidates(data, scenarios, call_index, base_for_llm, best,
                            replace(config.llm, timeout_seconds=timeout,
                                    max_retries=1, collect_diagnostics=True), context=context)
                    round_advice.append(advice)
                    timings["heuristic" if config.rule_advisor else "llm"] += time.perf_counter()-phase-advice.diagnostics.get("repair_seconds", 0.)
                    timings["repair"] += advice.diagnostics.get("repair_seconds", 0.)
                    diagnostics["llm_calls"].append(advice.diagnostics)
                    emit(f"{advisor_kind}_end", diagnostics=advice.diagnostics)
                    for candidate in advice.candidates:
                        if not available():
                            break
                        x = {key(kind, i, t):int(getattr(candidate, kind)[i,t]) for kind in ("u","v","w")
                             for i in range(data.n_units) for t in range(data.horizon)}
                        local_evaluate(x, 1, advisor_kind)
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
                if round_advice:
                    stalled = 0
            if not pool:
                break
            pool.sort(key=lambda r: r["cost"])
            pool = pool[:config.quantum.candidate_pool_size]
            pools.append(pool)
            lower_bounds.append(lb)
            previous_points[b] = [{k: result["point"][k] for k in block.bkeys} for result in pool]
            persistent[b] = cuts
        if len(pools) != len(blocks) or not available():
            break
        reliable_lower = max(reliable_lower, sum(lower_bounds))
        for rank in range(max(len(pool) for pool in pools)):
            if not available():
                break
            stitched = {k: v for pool in pools for k, v in pool[min(rank, len(pool)-1)]["point"].items()}
            u = np.array([[stitched[key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)], dtype=int)
            phase = time.perf_counter()
            candidate = repair_commitment(data, scenarios, u)
            if candidate is not None:
                candidate.source = "pool" if config.advisor_only else "sa"
            timings["repair"] += time.perf_counter()-phase
            diagnostics["proposals"].append({"outer": outer, "rank": rank, "raw_u": u,
                "hamming_u": int(np.count_nonzero(candidate.u != u)) if candidate else None,
                "accepted": candidate is not None})
            evaluate(candidate, outer)
        reported_lower = None if config.advisor_only else reliable_lower
        diagnostics["outer"].append({"outer": outer, "reliable_lower": reported_lower,
            "elapsed_seconds": time.perf_counter()-started})
        history.append(IterationRecord(outer, sum(len(p) for p in pools), 0, len(evaluated)-before_count,
            len(evaluated)-before_count, 0, optimality_count-before_opt, feasibility_count-before_feas, incumbent,
            llm_candidates=sum(advice.raw_count for advice in round_advice),
            llm_valid_candidates=sum(advice.valid_count for advice in round_advice),
            llm_failed=any(advice.failed for advice in round_advice)))
        emit("iteration", iteration=outer, incumbent=incumbent, reliable_lower=reported_lower, timings=dict(timings))
        if len(evaluated) == before_count and config.stop_on_no_new_candidates:
            status = "NO_NEW_CANDIDATES"
            break
    if config.target_cost is not None and incumbent < config.target_cost:
        status = "TARGET_REACHED"
    elif cancelled(config.llm):
        status = "CANCELLED"
    elif time.perf_counter() >= deadline:
        status = "TIME_LIMIT"
    elif config.max_candidate_evaluations is not None and len(evaluated) >= config.max_candidate_evaluations:
        status = "EVALUATION_LIMIT"
    if best is None:
        status = "NO_FEASIBLE_SOLUTION" if status not in {"TIME_LIMIT", "CANCELLED"} else status
    runtime = time.perf_counter()-started
    diagnostics.update(evaluation_count=len(evaluated), local_evaluation_count=total_local,
                       elapsed_seconds=runtime, time_limit_overshoot=max(0., time.perf_counter()-deadline),
                       reliable_lower=reliable_lower, block_count=len(blocks), coupling_count=0)
    diagnostics["sa_calls"] = sa_calls
    diagnostics["llm_call_count"] = 0 if config.rule_advisor else llm_calls
    diagnostics["rule_call_count"] = llm_calls if config.rule_advisor else 0
    if config.advisor_only:
        diagnostics["reliable_lower"] = None
    timings["other"] = max(0., runtime-sum(timings.values()))
    empty = np.empty((0, 0), dtype=int)
    return BendersResult(status, incumbent, best.u if best else empty, best.v if best else empty,
                         best.w if best else empty, best_solutions, history, optimality_count,
                         feasibility_count, runtime, diagnostics=diagnostics)
