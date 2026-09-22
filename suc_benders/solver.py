from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Sequence

import numpy as np

from .candidates import commitment_key, make_candidate, perturb_commitment, repair_commitment
from .common import Scenario, commitment_is_feasible
from .context import CandidateEvaluation, LocalMoveResult, LLMContextInput, build_llm_context
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
    perturb_candidates: bool = True
    perturbations_per_candidate: int = 12
    perturbation_window_lengths: tuple[int, ...] = (2, 3, 4)
    group_size: int = 2
    max_generated_candidates: int = 32
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    subproblem_output_flag: int = 0


@dataclass
class IterationRecord:
    iteration: int
    raw_candidates: int
    repaired_candidates: int
    perturbed_candidates: int
    feasible_candidates: int
    new_candidates: int
    feasible_subproblems: int
    optimality_cuts_added: int
    feasibility_cuts_added: int
    incumbent: float
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


def _fixed_cost(data: UCData, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return float(
        np.sum(data.cost_c[:, None] * u)
        + np.sum(data.startup_cost[:, None] * v)
        + np.sum(data.shutdown_cost[:, None] * w)
    )


def _construct_feasible_candidate(data: UCData, scenarios: Sequence[Scenario]) -> MasterCandidate:
    """Deterministic fallback: start every unit after its initial restriction."""

    n, horizon = data.n_units, data.horizon
    u = np.zeros((n, horizon), dtype=int)
    for i in range(n):
        start = 0 if data.initial_on[i] else max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
        u[i, start:] = 1
    candidate = make_candidate(data, scenarios, u, source="fallback")
    if not commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
        raise RuntimeError("deterministic fallback commitment is infeasible")
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
    started = time.perf_counter()

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

    def evaluate_candidate(candidate: MasterCandidate, source: str) -> CandidateEvaluation:
        nonlocal incumbent, best_u, best_v, best_w, best_scenario_results, best_solutions

        key = commitment_key(candidate.u, candidate.v, candidate.w)
        if key in evaluation_cache:
            return evaluation_cache[key]

        scenario_results = [
            solve_scenario(
                data,
                scenario,
                candidate.u,
                candidate.v,
                candidate.w,
                output_flag=config.subproblem_output_flag,
            )
            for scenario in scenarios
        ]
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
            for result in scenario_results:
                if not result.feasible:
                    cuts.append(BendersCut(result.cut_alpha, result.cut_beta_u, result.cut_beta_v, result.cut_beta_w, "feasibility"))
                    diagnosis = _diagnose_infeasible_result(source, result, scenario)
                    infeasible_diagnostics.append(diagnosis)
                    current_infeasible_diagnostics.append(diagnosis)
                    break

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
            feasibility_cuts=0 if feasible else 1,
            hamming_to_incumbent=hamming,
        )
        evaluation_cache[key] = evaluation
        return evaluation

    for iteration in range(1, config.max_iterations + 1):
        current_infeasible_diagnostics.clear()
        master = QuantumMaster(
            data,
            scenarios,
            cuts,
            theta_cap=theta_cap,
            penalty=config.penalty,
            theta_step=config.theta_step,
            reserve_scale=config.reserve_scale,
        )
        binaries = solve_qubo(master.matrix(), config.quantum)
        quantum_candidates = [master.decode(binary) for binary in binaries]
        fallback = _construct_feasible_candidate(data, scenarios)

        raw_count = len(quantum_candidates) + 1
        repaired_count = 0
        base_candidates: list[MasterCandidate] = []
        for candidate in quantum_candidates:
            if config.repair_candidates:
                repaired = repair_commitment(data, scenarios, candidate.u)
                if repaired is None:
                    current_infeasible_diagnostics.append({
                        "source": "quantum",
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
                base_candidates.append(candidate)
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

        # Evaluate base candidates first so local search and LLM context use
        # the best currently available incumbent.
        cache_size_at_iteration_start = len(evaluation_cache)
        cuts_size_at_iteration_start = len(cuts)
        base_evaluations: list[CandidateEvaluation] = []
        for candidate in unique_base:
            base_evaluations.append(evaluate_candidate(candidate, candidate.source))

        # Generate local neighbors around the current best incumbent and base
        # candidates, then evaluate them to obtain real cost changes.
        generated: list[MasterCandidate] = []
        perturbed_count = 0
        if config.perturb_candidates:
            for base in unique_base:
                neighbors = perturb_commitment(
                    data,
                    scenarios,
                    base,
                    limit=int(config.perturbations_per_candidate),
                    window_lengths=config.perturbation_window_lengths,
                    group_size=int(config.group_size),
                    seed=config.quantum.kaiwu_seed,
                )
                perturbed_count += len(neighbors)
                generated.extend(neighbors)

        local_evaluations: list[CandidateEvaluation] = []
        for candidate in generated[: int(config.max_generated_candidates)]:
            previous_incumbent = incumbent
            evaluation = evaluate_candidate(candidate, "local")
            local_evaluations.append(evaluation)
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
                        source="local",
                    )
                )
        local_move_history = local_move_history[-100:]

        # Build the compact P0/P1 context before asking the LLM for advice.
        llm_context = None
        if config.llm.enabled:
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
            )
            llm_context = build_llm_context(data, scenarios, context_input)

        llm_result = None
        llm_candidates: list[MasterCandidate] = []
        if config.llm.enabled:
            incumbent_candidate = None
            if best_u is not None:
                incumbent_candidate = make_candidate(data, scenarios, best_u, source="incumbent")
            llm_result = ask_llm_for_candidates(
                data,
                scenarios,
                iteration,
                unique_base,
                incumbent_candidate,
                config.llm,
                context=llm_context,
            )
            llm_candidates = llm_result.candidates

        llm_evaluations: list[CandidateEvaluation] = []
        for candidate in llm_candidates:
            evaluation = evaluate_candidate(candidate, "llm")
            llm_evaluations.append(evaluation)

        new_candidates = len(evaluation_cache) - cache_size_at_iteration_start
        feasible_subproblems = sum(
            1
            for evaluation in base_evaluations + local_evaluations + llm_evaluations
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
                raw_candidates=raw_count,
                repaired_candidates=repaired_count,
                perturbed_candidates=perturbed_count,
                feasible_candidates=len(unique_base) + len(generated) + len(llm_candidates),
                new_candidates=new_candidates,
                feasible_subproblems=feasible_subproblems,
                optimality_cuts_added=optimality_added,
                feasibility_cuts_added=feasibility_added,
                incumbent=incumbent,
                llm_candidates=llm_result.raw_count if llm_result else 0,
                llm_valid_candidates=llm_result.valid_count if llm_result else 0,
                llm_failed=llm_result.failed if llm_result else False,
                message=llm_result.message if llm_result else "",
            )
        )

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

    optimality_count = sum(cut.kind == "optimality" for cut in cuts)
    feasibility_count = sum(cut.kind == "feasibility" for cut in cuts)
    if best_u is None:
        return BendersResult(
            status="INFEASIBLE",
            objective=float("inf"),
            u=np.empty((data.n_units, data.horizon)),
            v=np.empty_like(np.empty((data.n_units, data.horizon))),
            w=np.empty_like(np.empty((data.n_units, data.horizon))),
            scenarios=[],
            history=history,
            optimality_cuts=optimality_count,
            feasibility_cuts=feasibility_count,
            runtime=time.perf_counter() - started,
            message="no candidate was feasible in every scenario",
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
    )


def result_to_dict(result: BendersResult) -> dict:
    def clean(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [clean(item) for item in value]
        if hasattr(value, "__dict__"):
            return {key: clean(val) for key, val in vars(value).items()}
        return value

    return clean(result)
