from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Sequence

import numpy as np

from .candidates import commitment_key, make_candidate, perturb_commitment, repair_commitment
from .common import Scenario, commitment_is_feasible
from .data import UCData
from .llm import LLMConfig, ask_llm_for_candidates
from .master import BendersCut, MasterCandidate, QuantumConfig, QuantumMaster
from .quantum import solve_qubo
from .subproblem import aggregate_optimality_cut, solve_scenario


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
    candidate = make_candidate(data, scenarios, u)
    if not commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
        raise RuntimeError("deterministic fallback commitment is infeasible")
    return candidate


def solve_benders(
    data: UCData,
    scenarios: list[Scenario],
    config: BendersConfig | None = None,
) -> BendersResult:
    config = config or BendersConfig()
    started = time.perf_counter()
    cuts: list[BendersCut] = []
    seen: set[tuple[int, ...]] = set()
    history: list[IterationRecord] = []
    incumbent: float = float("inf")
    best_u = best_v = best_w = None
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
    for iteration in range(1, config.max_iterations + 1):
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
                    continue
                if not np.array_equal(repaired.u, candidate.u):
                    repaired_count += 1
                base_candidates.append(repaired)
            elif commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
                base_candidates.append(candidate)
        base_candidates.append(fallback)
        if best_u is not None:
            base_candidates.append(make_candidate(data, scenarios, best_u))

        # Remove duplicate base candidates while preserving order.
        unique_base: list[MasterCandidate] = []
        base_seen: set[tuple[int, ...]] = set()
        for candidate in base_candidates:
            key = commitment_key(candidate.u, candidate.v, candidate.w)
            if key not in base_seen:
                base_seen.add(key)
                unique_base.append(candidate)

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

        llm_result = None
        llm_candidates: list[MasterCandidate] = []
        if config.llm.enabled:
            incumbent_candidate = None
            if best_u is not None:
                incumbent_candidate = make_candidate(data, scenarios, best_u)
            llm_result = ask_llm_for_candidates(
                data,
                scenarios,
                iteration,
                unique_base,
                incumbent_candidate,
                config.llm,
            )
            llm_candidates = llm_result.candidates

        # LLM candidates are scarce and expensive; let them survive the cap first.
        all_candidates = llm_candidates + generated + unique_base
        unique_candidates: list[MasterCandidate] = []
        candidate_seen: set[tuple[int, ...]] = set()
        for candidate in all_candidates:
            key = commitment_key(candidate.u, candidate.v, candidate.w)
            if key in candidate_seen:
                continue
            if not commitment_is_feasible(data, scenarios, candidate.u, candidate.v, candidate.w):
                continue
            candidate_seen.add(key)
            unique_candidates.append(candidate)
        unique_candidates = unique_candidates[: int(config.max_generated_candidates)]

        new_candidates: list[MasterCandidate] = []
        for candidate in unique_candidates:
            key = commitment_key(candidate.u, candidate.v, candidate.w)
            if key not in seen:
                seen.add(key)
                new_candidates.append(candidate)

        optimality_added = 0
        feasibility_added = 0
        feasible_subproblems = 0

        for candidate in new_candidates:
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
            if all(result.feasible for result in scenario_results):
                feasible_subproblems += 1
                expected_cost = sum(
                    scenario.probability * result.objective
                    for scenario, result in zip(scenarios, scenario_results)
                )
                total = _fixed_cost(data, candidate.u, candidate.v, candidate.w) + expected_cost
                if total < incumbent - 1e-7:
                    incumbent = total
                    best_u, best_v, best_w = candidate.u.copy(), candidate.v.copy(), candidate.w.copy()
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
                optimality_added += 1
            else:
                for result in scenario_results:
                    if not result.feasible:
                        cuts.append(BendersCut(result.cut_alpha, result.cut_beta_u, result.cut_beta_v, result.cut_beta_w, "feasibility"))
                        feasibility_added += 1
                        break

        history.append(
            IterationRecord(
                iteration=iteration,
                raw_candidates=raw_count,
                repaired_candidates=repaired_count,
                perturbed_candidates=perturbed_count,
                feasible_candidates=len(unique_candidates),
                new_candidates=len(new_candidates),
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
        if len(new_candidates) == 0:
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
