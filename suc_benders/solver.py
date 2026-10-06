from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Sequence

import numpy as np

from .candidates import make_candidate
from .common import Scenario, commitment_is_feasible
from .data import UCData
from .llm import LLMConfig
from .master import MasterCandidate, QuantumConfig


@dataclass
class BendersConfig:
    max_iterations: int = 20
    block_hours: int = 3
    max_inner_iterations: int = 20
    convergence_tolerance: float = 1e-4
    penalty: float = 1_000_000.0
    theta_step: float = 1.0
    reserve_scale: float = 100.0
    quantum_candidates: bool = True
    advisor_only: bool = False
    target_cost: float | None = None
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_candidate_evaluations: int | None = None
    time_limit_seconds: float | None = None
    stop_on_no_new_candidates: bool | None = None
    sa_seed_per_call: bool = False
    native_fallback_patience: int | None = None
    defer_full_recovery: bool = False
    gap_experiment_variant: str | None = None
    rule_advisor: bool = False
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


def solve_benders(
    data: UCData,
    scenarios: list[Scenario],
    config: BendersConfig | None = None,
) -> BendersResult:
    """Run temporal Benders/SA with certified Lagrangian coordination."""
    config = config or BendersConfig()
    if config.target_cost is not None or config.stop_on_no_new_candidates is not None:
        raise ValueError("target_cost and stop_on_no_new_candidates are retired; use convergence_tolerance and max_inner_iterations")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
           for v in (config.max_iterations, config.max_inner_iterations)):
        raise ValueError("max_iterations and max_inner_iterations must be positive finite integers")
    if type(config.block_hours) is not int or config.block_hours not in (1, 2, 3, 5):
        raise ValueError("block_hours must be 1, 2, 3 or 5")
    if not math.isfinite(config.convergence_tolerance) or config.convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive and finite")
    if config.rule_advisor and (config.llm.enabled or config.advisor_only):
        raise ValueError("rule advisor requires SA and LLM disabled")
    variants = ("sa_strict", "sa_flow", "sa_compact", "sa_full", "native_diagnostic")
    if config.gap_experiment_variant is not None:
        if config.gap_experiment_variant not in variants:
            raise ValueError("unknown pure-SA gap experiment variant")
        if config.llm.enabled or config.advisor_only or config.rule_advisor:
            raise ValueError("gap experiment variants disable all advisors")
        if config.gap_experiment_variant.startswith("sa_") and (
                not config.quantum_candidates or config.native_fallback_patience is not None):
            raise ValueError("pure-SA search forbids native integer candidates")
        if config.gap_experiment_variant == "native_diagnostic" and (
                config.quantum_candidates or config.native_fallback_patience != 1):
            raise ValueError("native diagnostic requires native-only candidates")
    if config.sa_seed_per_call and config.quantum.kaiwu_seed is None:
        raise ValueError("changing SA seeds requires a starting seed")
    if config.advisor_only and (not config.llm.enabled or config.quantum_candidates):
        raise ValueError("advisor_only requires LLM enabled and SA disabled")
    if config.native_fallback_patience is not None and (
            type(config.native_fallback_patience) is not int or config.native_fallback_patience < 1):
        raise ValueError("native_fallback_patience must be a positive integer or None")
    if type(config.defer_full_recovery) is not bool or (config.defer_full_recovery and
            (config.llm.enabled or config.advisor_only or config.rule_advisor)):
        raise ValueError("deferred full recovery requires a boolean flag and LLM/rule advisors disabled")
    if not config.quantum_candidates and not config.advisor_only and config.native_fallback_patience is None:
        raise ValueError("enable SA or select advisor_only")
    if config.time_limit_seconds is not None and (not math.isfinite(config.time_limit_seconds) or config.time_limit_seconds <= 0):
        raise ValueError("time limit must be positive and finite")
    if config.max_candidate_evaluations is not None and config.max_candidate_evaluations < 1:
        raise ValueError("max_candidate_evaluations must be positive or None")
    from .paper_solver import solve_paper
    return solve_paper(data, scenarios, config)
