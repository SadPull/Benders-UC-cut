"""SA-only integer search with bounded Kelley coordination of time blocks."""
from __future__ import annotations

from dataclasses import replace
import math
import time
from itertools import count

import numpy as np

from .candidates import repair_commitment, commitment_key, make_candidate
from .context import LLMContextInput, CandidateEvaluation, build_llm_context
from .llm import ask_llm_for_candidates
from .common import require_gurobi
from .paper_model import IndexedUC, Block, CompactQubo, key
from .quantum import solve_qubo
from .subproblem import solve_scenario


def coordinate(uc, planes, deadline):
    gp = require_gurobi()
    if not planes or time.perf_counter() >= deadline:
        return None
    m = gp.Model("paper_kelley_coordination")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(.001, deadline-time.perf_counter())
    prices = [m.addVar(lb=-10. if row.equality else 0., ub=10.) for row in uc.coupling]
    eta = m.addVar(lb=-gp.GRB.INFINITY)
    for cost, residual in planes:
        m.addConstr(eta <= cost/uc.cost_scale + gp.quicksum(float(r)*p for r, p in zip(residual, prices)))
    m.setObjective(eta, gp.GRB.MAXIMIZE)
    m.optimize()
    result = (np.array([p.X for p in prices]), float(m.ObjVal)*uc.cost_scale) if m.Status == gp.GRB.OPTIMAL else None
    m.dispose()
    return result


def solve_paper(data, scenarios, config):
    from .solver import BendersResult, IterationRecord, ScenarioSolution, _construct_feasible_candidate, _fixed_cost

    if (config.llm.enabled and (config.decomposition == "temporal" or config.master_encoding != "dynamic")) or config.decommit_candidates or config.random_candidates or not config.quantum_candidates:
        raise ValueError("paper modes require SA, with LLM/decommit/random disabled")
    if not config.repair_candidates:
        raise ValueError("paper modes require the configured full-horizon repair")
    started = time.perf_counter()
    deadline = started + (config.time_limit_seconds if config.time_limit_seconds is not None else float("inf"))
    if config.run_to_deadline and not math.isfinite(deadline):
        raise ValueError("run_to_deadline requires a finite time limit")
    sa_calls = 0

    def emit(kind, **payload):
        if config.event_sink is not None:
            config.event_sink({"kind": kind, "elapsed_seconds": time.perf_counter()-started,
                               "sa_calls": sa_calls, **payload})

    emit("started", deadline_clock=deadline)
    temporal = config.decomposition == "temporal"
    uc = IndexedUC(data, scenarios, config.block_hours if temporal else data.horizon)
    blocks = [Block(uc, j) for j in range(len(uc.blocks))]
    diagnostics = {"evaluations": [], "proposals": [], "sampling": [], "iterations": [],
                   "llm_calls": [], "outer": [], "local_evaluations": [],
                   "timings": {k: 0. for k in ("qubo", "annealing", "relaxation", "local_dispatch", "coordination", "repair", "subproblems", "llm", "context")}}
    timings = diagnostics["timings"]
    incumbent, best, best_solutions = float("inf"), None, []
    evaluated = set()
    history, planes = [], []
    multipliers = np.zeros(len(uc.coupling))
    optimality_count = feasibility_count = 0
    reliable_lower = -float("inf")
    total_local = 0
    persistent = [[] for _ in blocks]
    persistent_cache = [{} for _ in blocks]
    infeasible_cache = [set() for _ in blocks]
    previous_points = [[] for _ in blocks]
    best_results = []
    context_evaluations = []

    def available():
        return time.perf_counter() < deadline and (config.max_candidate_evaluations is None or len(evaluated) < config.max_candidate_evaluations)

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
                return False
            emit("dispatch_start", scope="full", scenario=scenario.name)
            results.append(solve_scenario(data, scenario, candidate.u, candidate.v, candidate.w,
                                         deadline=deadline, threads=1))
        elapsed = time.perf_counter()-phase
        timings["subproblems"] += elapsed
        # An interrupted solve is not a complete statistical evaluation.
        if time.perf_counter() >= deadline or any(r.status.startswith("TIME_LIMIT") or "TIME_LIMIT" in r.status for r in results):
            return False
        evaluated.add(signature)
        feasible = all(r.feasible for r in results)
        fixed = _fixed_cost(data, candidate.u, candidate.v, candidate.w)
        fuel = sum(s.probability*r.objective for s, r in zip(scenarios, results)) if feasible else float("inf")
        objective = fixed+fuel
        solutions = [ScenarioSolution(s.name, s.probability, r.objective, r.p, r.r) for s, r in zip(scenarios, results)] if feasible else []
        if feasible and config.event_sink is not None:
            from .attribution import validate_schedule
            from types import SimpleNamespace
            validate_schedule(data, scenarios, SimpleNamespace(objective=objective, u=candidate.u,
                              v=candidate.v, w=candidate.w, scenarios=solutions))
        if time.perf_counter() >= deadline:
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
            "commitment_id": str(signature), "u": candidate.u, "v": candidate.v, "w": candidate.w,
            "subproblem_seconds": elapsed, "sa_calls": sa_calls})
        emit("evaluation", **{k:v for k,v in diagnostics["evaluations"][-1].items() if k != "elapsed_seconds"},
             scenarios=solutions)
        return True

    seed = _construct_feasible_candidate(data, scenarios)
    evaluate(seed, 0)
    status = "MAX_ITERATIONS"
    for outer in (count(1) if config.run_to_deadline else range(1, config.max_iterations+1)):
        if not available():
            break
        costs, lag_constant = uc.coefficients(multipliers)
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
            # Integer constraints do not depend on prices; recourse objectives do.
            cuts = list(block.feasibility_cuts) if temporal else persistent[b]
            cache = {} if temporal else persistent_cache[b]
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
                    block.feasibility_cuts.append(result["cut"])
                    # This certificate depends on unchanged local constraints,
                    # not on the Lagrangian objective or its multiplier version.
                    infeasible_cache[b].add(signature)
                diagnostics["local_evaluations"].append({"outer": outer, "block": b, "inner": inner,
                    "multiplier_version": outer if temporal else 0, "origin": origin,
                    "feasible": result["feasible"], "cost": result["cost"] if result["feasible"] else None,
                    "local_evaluation_count": total_local, "elapsed_seconds": time.perf_counter()-started})

            local_evaluate({k: seed_point[k] for k in block.bkeys}, 0, "incumbent_projection")
            for point in previous_points[b]:
                local_evaluate(point, 0, "previous_sa")
            for inner in range(1, (config.block_inner_iterations if temporal else 1)+1):
                if not available():
                    break
                lp_value = None
                lp_seconds = 0.
                if config.master_encoding == "dynamic":
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
                                     lb if config.master_encoding == "dynamic" else analytic_lo,
                                     ub if config.master_encoding == "dynamic" else analytic_hi, config)
                build_seconds = time.perf_counter()-phase
                timings["qubo"] += build_seconds
                if not available():
                    break
                qconfig = config.quantum
                if temporal:
                    derived = int(np.random.SeedSequence([qconfig.kaiwu_seed or 0, outer, b, inner]).generate_state(1)[0])
                    qconfig = replace(qconfig, kaiwu_seed=derived)
                phase = time.perf_counter()
                sa_calls += 1
                emit("sa_start", iteration=outer, bits=len(master.names))
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
                    "lp_status": "OPTIMAL" if lp_value is not None else "UNAVAILABLE" if config.master_encoding == "dynamic" else "NOT_RUN",
                    "lp_seconds": lp_seconds, "build_seconds": build_seconds, "sampling_seconds": sampling_seconds,
                    "cuts": len(cuts), "elapsed_seconds": time.perf_counter()-started})
                for x in decoded:
                    if not temporal:
                        raw_u = np.array([[x[key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)])
                        phase = time.perf_counter()
                        repaired = repair_commitment(data, scenarios, raw_u)
                        timings["repair"] += time.perf_counter()-phase
                        diagnostics["proposals"].append({"outer": outer, "origin": "sa", "raw_u": raw_u,
                            "hamming_u": int(np.count_nonzero(repaired.u != raw_u)) if repaired else None,
                            "accepted": repaired is not None})
                        if repaired is None:
                            continue
                        x = {key(kind, i, t): int(getattr(repaired, kind)[i, t]) for kind in ("u", "v", "w")
                             for i in range(data.n_units) for t in range(data.horizon)}
                    local_evaluate(x, inner, "sa")
            if config.llm.enabled and available():
                # Complete SA evaluations first: LLM sees measured full-horizon feedback.
                base_for_llm = []
                for entry in sorted(pool, key=lambda r: r["cost"])[:config.quantum.candidate_pool_size]:
                    u = np.array([[entry["point"][key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)])
                    candidate = make_candidate(data, scenarios, u, source="sa")
                    evaluate(candidate, outer)
                    base_for_llm.append(candidate)
                if available():
                    phase = time.perf_counter()
                    context = build_llm_context(data, scenarios, LLMContextInput(
                        incumbent=best, incumbent_results=best_results, base_evaluations=context_evaluations[-16:],
                        local_move_results=[], candidate_evaluations=context_evaluations,
                        cuts=[], history=history))
                    context.pop("decommit_ladder", None)
                    context["block_cut_count"] = len(cuts)
                    timings["context"] += time.perf_counter()-phase
                    phase = time.perf_counter()
                    emit("llm_start", iteration=outer)
                    advice = ask_llm_for_candidates(data, scenarios, outer, base_for_llm, best,
                        replace(config.llm, timeout_seconds=max(.001, min(config.llm.timeout_seconds, deadline-time.perf_counter())),
                                max_retries=1, collect_diagnostics=True), context=context)
                    timings["llm"] += time.perf_counter()-phase-advice.diagnostics.get("repair_seconds", 0.)
                    timings["repair"] += advice.diagnostics.get("repair_seconds", 0.)
                    diagnostics["llm_calls"].append(advice.diagnostics)
                    emit("llm_end", diagnostics=advice.diagnostics)
                    for candidate in advice.candidates:
                        if not available():
                            break
                        x = {key(kind, i, t):int(getattr(candidate, kind)[i,t]) for kind in ("u","v","w")
                             for i in range(data.n_units) for t in range(data.horizon)}
                        local_evaluate(x, 1, "llm")
                        evaluate(candidate, outer)
            if not pool:
                break
            pool.sort(key=lambda r: r["cost"])
            pool = pool[:config.quantum.candidate_pool_size]
            pools.append(pool)
            lower_bounds.append(lb)
            previous_points[b] = [{k: result["point"][k] for k in block.bkeys} for result in pool]
            if not temporal:
                persistent[b] = cuts
        if len(pools) != len(blocks) or not available():
            break
        # Only complete sets at identical prices can be combined into a bound.
        reliable_lower = max(reliable_lower, sum(lower_bounds)+lag_constant)
        local_best = {k: v for pool in pools for k, v in pool[0]["point"].items()}
        residual = uc.residuals(local_best)
        original = uc.original_cost(local_best)
        planes.append((original, residual))
        identity = sum(pool[0]["cost"] for pool in pools)+lag_constant
        if not np.isclose(identity, original+uc.cost_scale*np.dot(multipliers, residual), rtol=1e-9, atol=1e-4):
            raise AssertionError("Lagrangian objective allocation mismatch")
        for rank in range(max(len(pool) for pool in pools)):
            if not available():
                break
            stitched = {k: v for pool in pools for k, v in pool[min(rank, len(pool)-1)]["point"].items()}
            u = np.array([[stitched[key("u", i, t)] for t in range(data.horizon)] for i in range(data.n_units)], dtype=int)
            phase = time.perf_counter()
            candidate = repair_commitment(data, scenarios, u)
            timings["repair"] += time.perf_counter()-phase
            raw_residual = uc.residuals(stitched)
            repaired_residual = None
            if candidate is not None:
                repaired_point = dict(stitched)
                repaired_point.update({key(kind, i, t): int(getattr(candidate, kind)[i, t]) for kind in ("u", "v", "w")
                                       for i in range(data.n_units) for t in range(data.horizon)})
                repaired_residual = uc.residuals(repaired_point)
            diagnostics["proposals"].append({"outer": outer, "rank": rank, "raw_u": u,
                "hamming_u": int(np.count_nonzero(candidate.u != u)) if candidate else None,
                "raw_coupling_residuals": raw_residual,
                "repaired_binary_with_raw_dispatch_residuals": repaired_residual,
                "accepted": candidate is not None})
            evaluate(candidate, outer)
        model_upper = None
        old_prices = multipliers.copy()
        if temporal and available():
            phase = time.perf_counter()
            update = coordinate(uc, planes, deadline)
            timings["coordination"] += time.perf_counter()-phase
            if update is not None:
                multipliers, model_upper = update
        diagnostics["outer"].append({"outer": outer, "reliable_lower": reliable_lower,
            "sampled_lagrangian_value": identity, "coordination_model_upper": model_upper,
            "normalized_residuals": residual, "prices": old_prices,
            "multiplier_boundary_fraction": float(np.mean(np.abs(multipliers) >= 10-1e-7)) if len(multipliers) else 0.,
            "elapsed_seconds": time.perf_counter()-started})
        history.append(IterationRecord(outer, sum(len(p) for p in pools), 0, len(evaluated)-before_count,
            len(evaluated)-before_count, 0, optimality_count-before_opt, feasibility_count-before_feas, incumbent))
        emit("iteration", iteration=outer, incumbent=incumbent, reliable_lower=reliable_lower, timings=dict(timings))
        if not temporal and len(evaluated) == before_count and not config.run_to_deadline:
            status = "NO_NEW_CANDIDATES"
            break
    if time.perf_counter() >= deadline:
        status = "TIME_LIMIT"
    elif config.max_candidate_evaluations is not None and len(evaluated) >= config.max_candidate_evaluations:
        status = "EVALUATION_LIMIT"
    if best is None:
        status = "NO_FEASIBLE_SOLUTION" if status != "TIME_LIMIT" else status
    runtime = time.perf_counter()-started
    diagnostics.update(evaluation_count=len(evaluated), local_evaluation_count=total_local,
                       elapsed_seconds=runtime, time_limit_overshoot=max(0., time.perf_counter()-deadline),
                       reliable_lower=reliable_lower, block_count=len(blocks), coupling_count=len(uc.coupling))
    diagnostics["sa_calls"] = sa_calls
    timings["other"] = max(0., runtime-sum(timings.values()))
    empty = np.empty((0, 0), dtype=int)
    return BendersResult(status, incumbent, best.u if best else empty, best.v if best else empty,
                         best.w if best else empty, best_solutions, history, optimality_count,
                         feasibility_count, runtime, diagnostics=diagnostics)
