from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Callable, List, Sequence

import numpy as np

from .candidates import (
    commitment_hash,
    commitment_key,
    decommit_commitment,
    make_candidate,
    repair_commitment,
)
from .common import Scenario, commitment_is_feasible
from .context import (
    CandidateEvaluation,
    LLMContextInput,
    LocalMoveResult,
    build_llm_context,
    decommit_gradient_map,
)
from .data import UCData
from .llm import LLMConfig, ask_llm_for_candidates
from .master import BendersCut, MasterCandidate, QuantumConfig, QuantumMaster
from .quantum import solve_qubo
from .subproblem import ScenarioResult, aggregate_optimality_cut, solve_scenario


@dataclass
class BendersConfig:
    max_iterations: int = 20
    penalty: float = 1_000_000.0
    theta_step: float = 1.0
    reserve_scale: float = 100.0
    repair_candidates: bool = True
    # Decommitment-first is the default direction: the gap is dominated by fixed
    # cost from over-committed peaking units.
    decommit_candidates: bool = True
    decommit_ladder_size: int = 8
    quantum_candidates: bool = True
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    subproblem_output_flag: int = 0
    max_candidate_evaluations: int | None = None
    collect_diagnostics: bool = False
    random_candidates: bool = False
    random_seed: int = 7
    random_candidate_pool_size: int = 16
    master_encoding: str = "legacy"
    decomposition: str = "none"
    block_hours: int = 2
    block_inner_iterations: int = 3
    time_limit_seconds: float | None = None
    # Used by the time-budget experiment; existing runs keep their stopping rules.
    run_to_deadline: bool = False
    event_sink: Callable[[dict], None] | None = field(default=None, repr=False)


@dataclass
class IterationRecord:
    iteration: int
    raw_candidates: int
    repaired_candidates: int
    feasible_candidates: int
    new_candidates: int
    feasible_subproblems: int
    optimality_cuts_added: int
    feasibility_cuts_added: int
    incumbent: float
    decommit_candidates: int = 0
    llm_candidates: int = 0
    llm_valid_candidates: int = 0
    llm_failed: bool = False
    message: str = ""


@dataclass
class ScenarioSolution:
    name: str
    probability: float
    objective: float
    p: np.ndarray
    r: np.ndarray


@dataclass
class BendersResult:
    status: str
    objective: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    scenarios: List[ScenarioSolution]
    history: List[IterationRecord]
    optimality_cuts: int
    feasibility_cuts: int
    runtime: float
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


def _fixed_cost(data: UCData, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return float(
        np.sum(data.cost_c[:, None] * u)
        + np.sum(data.startup_cost[:, None] * v)
        + np.sum(data.shutdown_cost[:, None] * w)
    )


def _construct_feasible_candidate(data: UCData, scenarios: Sequence[Scenario]) -> MasterCandidate | None:
    """Deterministic fallback: start every unit after its initial restriction."""

    n, horizon = data.n_units, data.horizon
    u = np.zeros((n, horizon), dtype=int)
    for i in range(n):
        start = 0 if data.initial_on[i] else max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
        u[i, start:] = 1
    candidate = make_candidate(data, scenarios, u, source="fallback")
    if not commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
        return None
    return candidate


def _diagnose_infeasible_result(source: str, result: ScenarioResult, scenario: Scenario) -> dict:
    return {
        "source": source,
        "type": result.status,
        "scenario": scenario.name,
        "unit": None,
        "period": None,
        "amount": None,
    }


def solve_benders(
    data: UCData,
    scenarios: list[Scenario],
    config: BendersConfig | None = None,
) -> BendersResult:
    config = config or BendersConfig()
    if config.master_encoding not in ("legacy", "compact", "dynamic") or config.decomposition not in ("none", "temporal"):
        raise ValueError("unknown encoding or decomposition")
    if config.block_hours < 1 or config.block_inner_iterations < 1:
        raise ValueError("block hours and inner iterations must be positive")
    if config.time_limit_seconds is not None and config.time_limit_seconds <= 0:
        raise ValueError("time limit must be positive")
    if config.decomposition == "temporal" and config.master_encoding != "dynamic":
        raise ValueError("temporal decomposition requires dynamic encoding")
    if config.master_encoding != "legacy" or config.decomposition != "none":
        from .paper_solver import solve_paper
        return solve_paper(data, scenarios, config)
    if config.max_candidate_evaluations is not None and config.max_candidate_evaluations < 1:
        raise ValueError("max_candidate_evaluations must be positive or None")
    if config.quantum_candidates and config.random_candidates:
        raise ValueError("random and quantum candidate generators are mutually exclusive")
    started = time.perf_counter()
    deadline = started + (config.time_limit_seconds if config.time_limit_seconds is not None else float("inf"))
    random_generator = np.random.default_rng(config.random_seed) if config.random_candidates else None
    timings = {name: 0.0 for name in ("qubo", "annealing", "random", "repair", "decommit", "context", "llm", "subproblems")}
    diagnostics = {
        "evaluations": [], "proposals": [], "sampling": [], "llm_calls": [],
        "iterations": [], "timings": timings,
    } if config.collect_diagnostics else {}

    cuts: list[BendersCut] = []
    history: list[IterationRecord] = []
    evaluation_cache: dict[tuple[int, ...], CandidateEvaluation] = {}
    local_move_history: list[LocalMoveResult] = []
    infeasible_diagnostics: list[dict] = []

    incumbent: float = float("inf")
    best_u: np.ndarray | None = None
    best_v: np.ndarray | None = None
    best_w: np.ndarray | None = None
    best_scenario_results: list[ScenarioResult] = []
    best_solutions: list[ScenarioSolution] = []

    optimality_count = 0
    feasibility_count = 0
    theta_cap = 1.0
    for scenario in scenarios:
        theta_cap += scenario.probability * float(
            np.sum(data.horizon * (data.cost_a * data.pmax**2 + data.cost_b * data.pmax))
        )

    status = "MAX_ITERATIONS"
    message = ""
    current_infeasible_diagnostics: list[dict] = []

    def budget_available() -> bool:
        return time.perf_counter() < deadline and (config.max_candidate_evaluations is None or len(evaluation_cache) < config.max_candidate_evaluations)

    def record_proposal(raw, final, origin, index, repair_attempted=False):
        proposal_id = f"{origin}:{iteration}:{index}"
        if final is not None:
            final.provenance = {
                "origin": origin, "proposal_id": proposal_id,
                "repair_attempted": repair_attempted,
                "repair_changed": commitment_key(raw.u, raw.v, raw.w) != commitment_key(final.u, final.v, final.w),
            }
        if not config.collect_diagnostics:
            return
        diagnostics["proposals"].append({
            "iteration": iteration, "proposal_id": proposal_id, "origin": origin,
            "raw_id": commitment_hash(raw), "raw_u": raw.u.copy(),
            "raw_v": raw.v.copy(), "raw_w": raw.w.copy(),
            "raw_discrete_feasible": commitment_is_feasible(data, scenarios, raw.u, raw.v, raw.w),
            "repair_attempted": repair_attempted,
            "commitment_id": commitment_hash(final) if final is not None else None,
            "repaired_u": final.u.copy() if final is not None else None,
            "hamming_u": int(np.count_nonzero(raw.u != final.u)) if final is not None else None,
            "hamming_uvw": sum(int(np.count_nonzero(a != b)) for a, b in zip(
                (raw.u, raw.v, raw.w), (final.u, final.v, final.w),
            )) if final is not None else None,
            "outcome": "accepted" if final is not None else "rejected",
        })

    def record_evaluation(candidate, source, evaluation, before, previous_u, cache_hit, elapsed):
        if not config.collect_diagnostics:
            return
        diagnostics["evaluations"].append({
            "iteration": iteration, "evaluation_count": len(evaluation_cache),
            "elapsed_seconds": time.perf_counter() - started,
            "commitment_id": commitment_hash(candidate),
            "source": source, "origin": candidate.provenance.get("origin", source),
            "provenance": dict(candidate.provenance), "cache_hit": cache_hit,
            "u": candidate.u.copy(), "v": candidate.v.copy(), "w": candidate.w.copy(),
            "feasible": evaluation.feasible, "scenario_costs": evaluation.scenario_costs,
            "scenario_feasible": {name: math.isfinite(value) for name, value in evaluation.scenario_costs.items()},
            "objective": evaluation.total_cost, "fixed_cost": evaluation.fixed_cost,
            "expected_fuel_cost": evaluation.expected_fuel_cost,
            "no_load_cost": float(np.sum(data.cost_c[:, None] * candidate.u)),
            "startup_cost": float(np.sum(data.startup_cost[:, None] * candidate.v)),
            "shutdown_cost": float(np.sum(data.shutdown_cost[:, None] * candidate.w)),
            "incumbent_before": before, "incumbent": incumbent,
            "improved": bool(not cache_hit and incumbent < before - 1e-7),
            "improvement": before - incumbent if math.isfinite(before) else None,
            "hamming_to_previous_incumbent": int(np.count_nonzero(candidate.u != previous_u)) if previous_u is not None else None,
            "subproblem_seconds": elapsed,
            "scenario_solves": 0 if cache_hit else len(scenarios),
        })

    def evaluate_candidate(candidate: MasterCandidate, source: str) -> CandidateEvaluation | None:
        nonlocal incumbent, best_u, best_v, best_w, best_scenario_results, best_solutions

        key = commitment_key(candidate.u, candidate.v, candidate.w)
        if key in evaluation_cache:
            record_evaluation(candidate, source, evaluation_cache[key], incumbent, best_u, True, 0.0)
            return evaluation_cache[key]
        if not budget_available():
            return None

        before = incumbent
        previous_u = best_u
        scenario_started = time.perf_counter()
        scenario_results = [
            solve_scenario(
                data,
                scenario,
                candidate.u,
                candidate.v,
                candidate.w,
                output_flag=config.subproblem_output_flag,
                **({"deadline": deadline, "threads": 1} if config.time_limit_seconds is not None else {}),
            )
            for scenario in scenarios
        ]
        scenario_seconds = time.perf_counter() - scenario_started
        timings["subproblems"] += scenario_seconds
        if any("TIME_LIMIT" in result.status for result in scenario_results):
            return None
        feasible = all(result.feasible for result in scenario_results)
        fixed = _fixed_cost(data, candidate.u, candidate.v, candidate.w)
        expected_fuel = (
            sum(scenario.probability * result.objective for scenario, result in zip(scenarios, scenario_results))
            if feasible
            else float("inf")
        )
        total = fixed + expected_fuel

        if feasible:
            if total < incumbent - 1e-7:
                incumbent = total
                best_u, best_v, best_w = candidate.u.copy(), candidate.v.copy(), candidate.w.copy()
                best_scenario_results = [result for result in scenario_results]
                best_solutions = [
                    ScenarioSolution(scenario.name, scenario.probability, result.objective, result.p.copy(), result.r.copy())
                    for scenario, result in zip(scenarios, scenario_results)
                ]
            aggregated = aggregate_optimality_cut(scenario_results, scenarios)
            cuts.append(BendersCut(
                aggregated.cut_alpha,
                aggregated.cut_beta_u,
                aggregated.cut_beta_v,
                aggregated.cut_beta_w,
                "optimality",
            ))
        else:
            for scenario, result in zip(scenarios, scenario_results):
                if not result.feasible:
                    cuts.append(BendersCut(result.cut_alpha, result.cut_beta_u, result.cut_beta_v, result.cut_beta_w, "feasibility"))
                    diagnosis = _diagnose_infeasible_result(source, result, scenario)
                    infeasible_diagnostics.append(diagnosis)
                    current_infeasible_diagnostics.append(diagnosis)

        hamming = (
            int(np.count_nonzero(candidate.u != best_u)) if best_u is not None else None
        )
        evaluation = CandidateEvaluation(
            source=source,
            total_cost=total,
            fixed_cost=fixed,
            expected_fuel_cost=expected_fuel,
            scenario_costs={
                scenario.name: result.objective if result.feasible else float("inf")
                for scenario, result in zip(scenarios, scenario_results)
            },
            feasible=feasible,
            optimality_cuts=1 if feasible else 0,
            feasibility_cuts=sum(not result.feasible for result in scenario_results),
            hamming_to_incumbent=hamming,
        )
        evaluation_cache[key] = evaluation
        record_evaluation(candidate, source, evaluation, before, previous_u, False, scenario_seconds)
        return evaluation

    for iteration in range(1, config.max_iterations + 1):
        if not budget_available():
            break
        iteration_started = time.perf_counter()
        current_infeasible_diagnostics.clear()

        quantum_candidates = []
        if config.quantum_candidates:
            phase_started = time.perf_counter()
            master = QuantumMaster(
                data,
                scenarios,
                cuts,
                theta_cap=theta_cap,
                penalty=config.penalty,
                theta_step=config.theta_step,
                reserve_scale=config.reserve_scale,
            )
            matrix = master.matrix()
            build_seconds = time.perf_counter() - phase_started
            timings["qubo"] += build_seconds
            if time.perf_counter() >= deadline:
                break
            phase_started = time.perf_counter()
            binaries = solve_qubo(matrix, config.quantum, energy_unit=master.energy_unit)
            sampling_seconds = time.perf_counter() - phase_started
            timings["annealing"] += sampling_seconds
            quantum_candidates = [master.decode(binary) for binary in binaries]
            if config.collect_diagnostics:
                diagnostics["sampling"].append({
                    "iteration": iteration, "cuts": len(cuts), "bits": int(matrix.shape[0]),
                    "decision_bits": master.variable_count(),
                    "auxiliary_bits": int(matrix.shape[0])-master.variable_count(),
                    "density": float(np.count_nonzero(matrix)/matrix.size) if matrix.size else 0.,
                    "max_abs_coefficient": float(np.max(np.abs(matrix))),
                    "energy_unit": master.energy_unit, "build_seconds": build_seconds,
                    "sampling_seconds": sampling_seconds, "raw_candidates": len(quantum_candidates),
                    "unique_u": len({tuple(candidate.u.ravel()) for candidate in quantum_candidates}),
                    "unique_uvw": len({commitment_key(candidate.u, candidate.v, candidate.w) for candidate in quantum_candidates}),
                    "raw_discrete_feasible": sum(commitment_is_feasible(data, scenarios, c.u, c.v, c.w) for c in quantum_candidates),
                })
        if random_generator is not None:
            phase_started = time.perf_counter()
            quantum_candidates = [make_candidate(
                data, scenarios, random_generator.integers(0, 2, size=(data.n_units, data.horizon)),
                source="random",
            ) for _ in range(config.random_candidate_pool_size)]
            timings["random"] += time.perf_counter() - phase_started

        fallback = _construct_feasible_candidate(data, scenarios)

        raw_candidates = len(quantum_candidates) + int(fallback is not None)
        repaired_count = 0
        base_candidates: list[MasterCandidate] = []
        for index, candidate in enumerate(quantum_candidates):
            origin = candidate.source
            if config.repair_candidates:
                phase_started = time.perf_counter()
                repaired = repair_commitment(data, scenarios, candidate.u)
                timings["repair"] += time.perf_counter() - phase_started
                record_proposal(candidate, repaired, origin, index, True)
                if repaired is None:
                    current_infeasible_diagnostics.append({
                        "source": origin,
                        "type": "repair_failed",
                        "unit": None,
                        "period": None,
                        "amount": None,
                    })
                    continue
                if not np.array_equal(repaired.u, candidate.u):
                    repaired_count += 1
                base_candidates.append(repaired)
            elif commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
                record_proposal(candidate, candidate, origin, index)
                base_candidates.append(candidate)
            else:
                record_proposal(candidate, None, origin, index)
        if fallback is not None:
            base_candidates.append(fallback)
        if best_u is not None:
            incumbent_candidate = make_candidate(data, scenarios, best_u, source="incumbent")
            base_candidates.append(incumbent_candidate)

        # De-duplicate base candidates.
        unique_base: list[MasterCandidate] = []
        base_seen: set[tuple[int, ...]] = set()
        for candidate in base_candidates:
            key = commitment_key(candidate.u, candidate.v, candidate.w)
            if key not in base_seen:
                base_seen.add(key)
                unique_base.append(candidate)
            elif config.collect_diagnostics:
                diagnostics["proposals"].append({
                    "iteration": iteration, "origin": candidate.provenance.get("origin", candidate.source),
                    "proposal_id": candidate.provenance.get("proposal_id"),
                    "commitment_id": commitment_hash(candidate), "outcome": "duplicate_in_base",
                })

        # Evaluate base candidates first so decommitment and LLM context use
        # the best currently available incumbent.
        cache_size_at_iteration_start = len(evaluation_cache)
        cuts_size_at_iteration_start = len(cuts)
        base_evaluations: list[CandidateEvaluation] = []
        for candidate in unique_base:
            evaluation = evaluate_candidate(candidate, candidate.source)
            if evaluation is None:
                break
            base_evaluations.append(evaluation)

        # Decommitment-first search: start from the incumbent and greedily turn
        # OFF on-hours, scored by the same per-(unit, hour) gradient the LLM
        # sees.
        decommit_evaluations: list[CandidateEvaluation] = []
        decommit_generated: list[MasterCandidate] = []
        if config.decommit_candidates and best_u is not None and budget_available():
            phase_started = time.perf_counter()
            seed_candidate = make_candidate(data, scenarios, best_u, source="incumbent")
            gradient = decommit_gradient_map(
                data, scenarios, seed_candidate, best_scenario_results
            )
            decommit_generated = decommit_commitment(
                data,
                scenarios,
                seed_candidate,
                limit=int(config.decommit_ladder_size),
                gradient=gradient,
            )
            timings["decommit"] += time.perf_counter() - phase_started
            for index, candidate in enumerate(decommit_generated):
                record_proposal(candidate, candidate, "decommit", index)
                candidate.provenance["parent_id"] = commitment_hash(seed_candidate)
                previous_incumbent = incumbent
                evaluation = evaluate_candidate(candidate, "decommit")
                if evaluation is None:
                    break
                decommit_evaluations.append(evaluation)
                if candidate.move is not None:
                    move = candidate.move
                    local_move_history.append(
                        LocalMoveResult(
                            move=str(move.get("move", "unknown")),
                            unit=int(move.get("unit", -1)),
                            period=int(move.get("period", -1)),
                            width=int(move.get("width", 1)),
                            cost_change=evaluation.total_cost - previous_incumbent,
                            improved=evaluation.total_cost < previous_incumbent,
                            source="decommit",
                        )
                    )
            local_move_history = local_move_history[-100:]

        # Build the compact P0/P1 context before asking the LLM for advice.
        llm_context = None
        if config.llm.enabled and budget_available():
            phase_started = time.perf_counter()
            incumbent_candidate = None
            if best_u is not None:
                incumbent_candidate = make_candidate(data, scenarios, best_u, source="incumbent")
            context_input = LLMContextInput(
                incumbent=incumbent_candidate,
                incumbent_results=best_scenario_results,
                base_evaluations=base_evaluations,
                local_move_results=local_move_history,
                candidate_evaluations=list(evaluation_cache.values()),
                cuts=cuts,
                history=history,
                infeasible_diagnostics=current_infeasible_diagnostics,
                decommit_ladder=decommit_evaluations,
            )
            llm_context = build_llm_context(data, scenarios, context_input)
            timings["context"] += time.perf_counter() - phase_started

        llm_result = None
        llm_candidates: list[MasterCandidate] = []
        if config.llm.enabled and budget_available():
            incumbent_candidate = None
            if best_u is not None:
                incumbent_candidate = make_candidate(data, scenarios, best_u, source="incumbent")
            phase_started = time.perf_counter()
            llm_result = ask_llm_for_candidates(
                data,
                scenarios,
                iteration,
                unique_base,
                incumbent_candidate,
                replace(config.llm, collect_diagnostics=True) if config.collect_diagnostics else config.llm,
                context=llm_context,
            )
            llm_seconds = time.perf_counter() - phase_started
            llm_repair_seconds = llm_result.diagnostics.get("repair_seconds", 0.0)
            timings["llm"] += max(0.0, llm_seconds - llm_repair_seconds)
            timings["repair"] += llm_repair_seconds
            if config.collect_diagnostics:
                call = dict(llm_result.diagnostics)
                call.update(iteration=iteration, start_seconds=phase_started - started,
                            end_seconds=time.perf_counter() - started, failed=llm_result.failed)
                diagnostics["llm_calls"].append(call)
                diagnostics["proposals"].extend({"iteration": iteration, **row} for row in call.get("proposals", []))
            llm_candidates = llm_result.candidates

        llm_evaluations: list[CandidateEvaluation] = []
        for candidate in llm_candidates:
            evaluation = evaluate_candidate(candidate, "llm")
            if evaluation is None:
                break
            llm_evaluations.append(evaluation)

        new_candidates = len(evaluation_cache) - cache_size_at_iteration_start
        feasible_subproblems = sum(
            1
            for evaluation in base_evaluations
            + decommit_evaluations
            + llm_evaluations
            if evaluation.feasible
        )
        optimality_added = sum(
            cut.kind == "optimality" for cut in cuts[cuts_size_at_iteration_start:]
        )
        feasibility_added = sum(
            cut.kind == "feasibility" for cut in cuts[cuts_size_at_iteration_start:]
        )

        history.append(
            IterationRecord(
                iteration=iteration,
                raw_candidates=raw_candidates,
                repaired_candidates=repaired_count,
                feasible_candidates=(
                    len(unique_base) + len(decommit_generated) + len(llm_candidates)
                ),
                new_candidates=new_candidates,
                feasible_subproblems=feasible_subproblems,
                optimality_cuts_added=optimality_added,
                feasibility_cuts_added=feasibility_added,
                incumbent=incumbent,
                decommit_candidates=len(decommit_evaluations),
                llm_candidates=llm_result.raw_count if llm_result else 0,
                llm_valid_candidates=llm_result.valid_count if llm_result else 0,
                llm_failed=llm_result.failed if llm_result else False,
                message=llm_result.message if llm_result else "",
            )
        )

        if config.collect_diagnostics:
            diagnostics["iterations"].append({
                "iteration": iteration, "elapsed_seconds": time.perf_counter() - started,
                "iteration_seconds": time.perf_counter() - iteration_started,
                "evaluation_count": len(evaluation_cache), "incumbent": incumbent,
                "timings": dict(timings),
            })
        if not budget_available():
            status = "TIME_LIMIT" if time.perf_counter() >= deadline else "EVALUATION_LIMIT"
            message = "unique candidate evaluation budget exhausted"
            break
        if new_candidates == 0:
            # A failed LLM call is a transient infrastructure event, not
            # evidence that candidate generation has genuinely stalled. Only
            # treat the iteration as a deadlock when the LLM did not fail.
            llm_failure = bool(llm_result and llm_result.failed)
            if llm_failure and iteration < config.max_iterations:
                continue
            status = "NO_NEW_CANDIDATES"
            message = "candidate generator produced no new discrete-feasible commitments"
            break
        if iteration < config.max_iterations:
            theta_cap = min(theta_cap, max(incumbent, 1.0))

    if time.perf_counter() >= deadline:
        status = "TIME_LIMIT"
    optimality_count = sum(cut.kind == "optimality" for cut in cuts)
    feasibility_count = sum(cut.kind == "feasibility" for cut in cuts)
    if config.collect_diagnostics:
        diagnostics["evaluation_count"] = len(evaluation_cache)
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        diagnostics["timings"]["other"] = max(0.0, diagnostics["elapsed_seconds"] - sum(timings.values()))
    if best_u is None:
        status = status if status == "EVALUATION_LIMIT" else "NO_FEASIBLE_SOLUTION"
        message = "no candidate was feasible in every scenario"
        return BendersResult(
            status=status,
            objective=float("inf"),
            u=np.empty((0, 0), dtype=int),
            v=np.empty((0, 0), dtype=int),
            w=np.empty((0, 0), dtype=int),
            scenarios=[],
            history=history,
            optimality_cuts=optimality_count,
            feasibility_cuts=feasibility_count,
            runtime=time.perf_counter() - started,
            message=message,
            diagnostics=diagnostics,
        )
    return BendersResult(
        status=status,
        objective=incumbent,
        u=best_u,
        v=best_v,
        w=best_w,
        scenarios=best_solutions,
        history=history,
        optimality_cuts=optimality_count,
        feasibility_cuts=feasibility_count,
        runtime=time.perf_counter() - started,
        message=message,
        diagnostics=diagnostics,
    )


def result_to_dict(result: BendersResult | dict) -> dict:
    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if hasattr(value, "__dict__"):
            return {key: clean(val) for key, val in vars(value).items()}
        return value

    return clean(result)
