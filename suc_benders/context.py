from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .common import Scenario
from .data import UCData
from .master import BendersCut, MasterCandidate
from .subproblem import ScenarioResult


@dataclass
class CandidateEvaluation:
    source: str
    total_cost: float
    fixed_cost: float
    expected_fuel_cost: float
    scenario_costs: dict[str, float]
    feasible: bool
    optimality_cuts: int
    feasibility_cuts: int
    hamming_to_incumbent: int | None


@dataclass
class LocalMoveResult:
    move: str
    unit: int
    period: int
    width: int
    cost_change: float
    improved: bool
    source: str = "local"


@dataclass
class LLMContextInput:
    incumbent: MasterCandidate | None
    incumbent_results: Sequence[ScenarioResult]
    base_evaluations: Sequence[CandidateEvaluation]
    local_move_results: Sequence[LocalMoveResult]
    candidate_evaluations: Sequence[CandidateEvaluation]
    cuts: Sequence[BendersCut]
    history: Sequence[Any]
    infeasible_diagnostics: Sequence[Mapping[str, Any]] = field(default_factory=list)


def _round3(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round3(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round3(item) for item in value]
    if isinstance(value, tuple):
        return [_round3(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 3) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _fixed_cost(data: UCData, incumbent: MasterCandidate) -> float:
    return float(
        np.sum(data.cost_c[:, None] * incumbent.u)
        + np.sum(data.startup_cost[:, None] * incumbent.v)
        + np.sum(data.shutdown_cost[:, None] * incumbent.w)
    )


def _scenario_costs(results: Sequence[ScenarioResult], scenarios: Sequence[Scenario]) -> dict[str, float]:
    return {
        scenario.name: float(result.objective)
        for scenario, result in zip(scenarios, results)
    }


def _expected_fuel(results: Sequence[ScenarioResult], scenarios: Sequence[Scenario]) -> float:
    return float(
        sum(scenario.probability * result.objective for scenario, result in zip(scenarios, results))
    )


def _base_candidate_rows(
    evaluations: Sequence[CandidateEvaluation],
    incumbent_total: float | None,
) -> list[dict]:
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: item.total_cost)[:10]:
        rows.append(
            {
                "id": len(rows),
                "source": evaluation.source,
                "total_cost": evaluation.total_cost,
                "fixed_cost": evaluation.fixed_cost,
                "expected_fuel_cost": evaluation.expected_fuel_cost,
                "better_than_incumbent": (
                    incumbent_total is not None and evaluation.total_cost < incumbent_total
                ),
                "hamming_to_incumbent": evaluation.hamming_to_incumbent,
            }
        )
    return rows


def _scenario_duals(results: Sequence[ScenarioResult]) -> dict:
    balance = [
        result.balance_dual.tolist() if result.balance_dual is not None else []
        for result in results
    ]
    reserve = [
        result.reserve_dual.tolist() if result.reserve_dual is not None else []
        for result in results
    ]
    return {"balance": balance, "reserve": reserve}


def _benders_cut_summary(
    cuts: Sequence[BendersCut],
    incumbent: MasterCandidate | None,
    expected_fuel: float | None,
) -> dict:
    optimality_cuts = [cut for cut in cuts if cut.kind == "optimality"]
    recent = optimality_cuts[-5:]
    tight_count = 0
    if incumbent is not None and expected_fuel is not None:
        for cut in optimality_cuts:
            value = cut.value(incumbent.u, incumbent.v, incumbent.w)
            if abs(value - expected_fuel) <= 1e-3:
                tight_count += 1
    if optimality_cuts:
        beta_u = np.sum([cut.beta_u for cut in optimality_cuts], axis=0)
        unit_scores = np.sum(np.abs(beta_u), axis=1)
        period_scores = np.sum(np.abs(beta_u), axis=0)
        top_units = np.argsort(-unit_scores)[:5].tolist()
        top_periods = np.argsort(-period_scores)[:5].tolist()
    else:
        top_units = []
        top_periods = []
    return {
        "total_cuts": len(cuts),
        "recent_cuts": len(recent),
        "tight_at_incumbent": tight_count,
        "top_covered_units": top_units,
        "top_covered_periods": top_periods,
    }


def _local_move_rows(results: Sequence[LocalMoveResult]) -> list[dict]:
    rows = []
    for result in sorted(results, key=lambda item: abs(item.cost_change), reverse=True)[:10]:
        rows.append(
            {
                "move": result.move,
                "unit": result.unit,
                "period": result.period,
                "width": result.width,
                "cost_change": result.cost_change,
                "improved": result.improved,
            }
        )
    return rows


def _candidate_comparison_rows(
    evaluations: Sequence[CandidateEvaluation],
) -> list[dict]:
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: item.total_cost)[:10]:
        rows.append(
            {
                "source": evaluation.source,
                "total_cost": evaluation.total_cost,
                "fixed_cost": evaluation.fixed_cost,
                "expected_fuel_cost": evaluation.expected_fuel_cost,
                "hamming_to_incumbent": evaluation.hamming_to_incumbent,
                "feasible": evaluation.feasible,
                "optimality_cuts": evaluation.optimality_cuts,
            }
        )
    return rows


def _scenario_dispatch_summary(
    data: UCData,
    scenarios: Sequence[Scenario],
    incumbent: MasterCandidate | None,
    results: Sequence[ScenarioResult],
) -> list[dict]:
    rows = []
    for scenario, result in zip(scenarios, results):
        p = np.asarray(result.p, dtype=float) if result.p is not None else None
        r = np.asarray(result.r, dtype=float) if result.r is not None else None
        marginal_count = 0
        if p is not None and incumbent is not None:
            at_min = np.isclose(p, data.pmin[:, None], atol=1e-5)
            at_max = np.isclose(p, data.pmax[:, None], atol=1e-5)
            marginal_count = int(np.sum((at_min | at_max) & (incumbent.u == 1)))
        rows.append(
            {
                "scenario": scenario.name,
                "total_generation": float(np.sum(p)) if p is not None else 0.0,
                "total_reserve": float(np.sum(r)) if r is not None else 0.0,
                "startup_count": int(np.sum(incumbent.v)) if incumbent is not None else 0,
                "shutdown_count": int(np.sum(incumbent.w)) if incumbent is not None else 0,
                "marginal_unit_count": marginal_count,
            }
        )
    return rows


def _reserve_margin(
    data: UCData,
    scenarios: Sequence[Scenario],
    incumbent: MasterCandidate | None,
) -> list[list[float]]:
    if incumbent is None:
        return [[] for _ in scenarios]
    capacity = (
        data.pmax[:, None] * incumbent.u
        - (data.pmax - data.pstart)[:, None] * incumbent.v
    )
    capacity[:, :-1] -= (data.pmax - data.pshut)[:, None] * incumbent.w[:, 1:]
    capacity = capacity.sum(axis=0)
    return [
        (capacity - (1.0 + data.reserve_fraction) * scenario.load).tolist()
        for scenario in scenarios
    ]


def _tight_constraints(
    scenarios: Sequence[Scenario],
    results: Sequence[ScenarioResult],
) -> list[dict]:
    rows = []
    for scenario, result in zip(scenarios, results):
        if result.balance_dual is not None:
            for period, dual in enumerate(result.balance_dual):
                rows.append(
                    {
                        "type": "balance",
                        "scenario": scenario.name,
                        "period": period,
                        "dual": float(dual),
                    }
                )
        if result.reserve_dual is not None:
            for period, dual in enumerate(result.reserve_dual):
                rows.append(
                    {
                        "type": "reserve",
                        "scenario": scenario.name,
                        "period": period,
                        "dual": float(dual),
                    }
                )
    rows.sort(key=lambda row: abs(row["dual"]), reverse=True)
    return rows[:20]


def _iteration_history_rows(history: Sequence[Any]) -> list[dict]:
    rows = []
    previous = None
    for record in list(history)[-5:]:
        incumbent = float(record.incumbent)
        improvement = 0.0 if previous is None else max(0.0, previous - incumbent)
        rows.append(
            {
                "iteration": int(record.iteration),
                "incumbent": incumbent,
                "improvement": improvement,
                "optimality_cuts": int(record.optimality_cuts_added),
                "feasibility_cuts": int(record.feasibility_cuts_added),
                "new_candidates": int(record.new_candidates),
                "llm_failures": 1 if record.llm_failed else 0,
            }
        )
        previous = incumbent
    return rows


def _unit_marginal_cost(
    data: UCData,
    scenarios: Sequence[Scenario],
    results: Sequence[ScenarioResult],
) -> list[dict]:
    if not results:
        return []
    expected_p = sum(
        scenario.probability * np.asarray(result.p, dtype=float)
        for scenario, result in zip(scenarios, results)
    )
    current_p = expected_p.mean(axis=1)
    rows = []
    for i in range(data.n_units):
        rows.append(
            {
                "unit": i,
                "at_pmin": float(2.0 * data.cost_a[i] * data.pmin[i] + data.cost_b[i]),
                "at_current_p": float(2.0 * data.cost_a[i] * current_p[i] + data.cost_b[i]),
                "at_pmax": float(2.0 * data.cost_a[i] * data.pmax[i] + data.cost_b[i]),
            }
        )
    return rows


def _unit_cost_decomposition(
    data: UCData,
    scenarios: Sequence[Scenario],
    incumbent: MasterCandidate | None,
    results: Sequence[ScenarioResult],
) -> list[dict]:
    if incumbent is None or not results:
        return []
    expected_p = sum(
        scenario.probability * np.asarray(result.p, dtype=float)
        for scenario, result in zip(scenarios, results)
    )
    current_p = expected_p.mean(axis=1)
    rows = []
    for i in range(data.n_units):
        fuel = float(
            np.sum(data.cost_a[i] * expected_p[i] ** 2 + data.cost_b[i] * expected_p[i])
        )
        rows.append(
            {
                "unit": i,
                "fixed_cost": float(np.sum(data.cost_c[i] * incumbent.u[i])),
                "startup_cost": float(np.sum(data.startup_cost[i] * incumbent.v[i])),
                "fuel_cost": fuel,
            }
        )
    return rows


def build_llm_context(
    data: UCData,
    scenarios: Sequence[Scenario],
    context_input: LLMContextInput,
) -> dict:
    """Build the compact P0/P1 context injected into the LLM prompt."""

    incumbent = context_input.incumbent
    results = context_input.incumbent_results
    fixed = _fixed_cost(data, incumbent) if incumbent is not None else None
    expected_fuel = _expected_fuel(results, scenarios) if results else None
    total = fixed + expected_fuel if fixed is not None and expected_fuel is not None else None
    incumbent_cost = None
    if total is not None:
        incumbent_cost = {
            "total": total,
            "fixed": fixed,
            "expected_fuel": expected_fuel,
            "scenario_fuel": _scenario_costs(results, scenarios),
        }

    return _round3(
        {
            "incumbent_cost": incumbent_cost,
            "base_candidates": _base_candidate_rows(
                context_input.base_evaluations, total
            ),
            "scenario_duals": _scenario_duals(results),
            "benders_cut_summary": _benders_cut_summary(
                context_input.cuts, incumbent, expected_fuel
            ),
            "local_move_results": _local_move_rows(context_input.local_move_results),
            "candidate_comparison": _candidate_comparison_rows(
                context_input.candidate_evaluations
            ),
            "scenario_dispatch_summary": _scenario_dispatch_summary(data, scenarios, incumbent, results),
            "reserve_margin": _reserve_margin(data, scenarios, incumbent),
            "tight_constraints": _tight_constraints(scenarios, results),
            "infeasible_diagnostics": list(context_input.infeasible_diagnostics)[:20],
            "iteration_history": _iteration_history_rows(context_input.history),
            "unit_marginal_cost": _unit_marginal_cost(data, scenarios, results),
            "unit_cost_decomposition": _unit_cost_decomposition(
                data, scenarios, incumbent, results
            ),
        }
    )
