"""Small strategy seams for constraint encoding, cut selection and sampling."""
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .anneal import anneal
from .constraints import cover_policy, legacy_square_policy


class ConstraintPolicy(Protocol):
    name: str

    def rows(self, b, cuts, options, extra_rows=()): ...

    def observe(self, b, candidate, reason): ...

    def weight(self, kind, local_bound, options): ...


CutSelector = Callable[..., tuple[list[int], list[int]]]
SamplerBackend = Callable[..., tuple[list[np.ndarray], dict]]


def recent_tight_selector(cuts, values, columns, optimality_limit, feasibility_limit):
    optimality = [k for k, cut in enumerate(cuts) if cut.optimality]
    feasibility = [k for k, cut in enumerate(cuts) if not cut.optimality]
    chosen = []
    if optimality:
        anchor = max(optimality, key=lambda k: values[k])
        chosen = [anchor] + [k for k in reversed(optimality) if k != anchor][:optimality_limit-1]
    selected_feasibility = sorted(feasibility, key=lambda k: -values[k])[:feasibility_limit]
    return chosen, selected_feasibility


def coverage_selector(cuts, values, columns, optimality_limit, feasibility_limit):
    """Prefer cuts with a different block slope after retaining the tightest cut."""
    optimality = [k for k, cut in enumerate(cuts) if cut.optimality]
    feasibility = [k for k, cut in enumerate(cuts) if not cut.optimality]
    chosen = []
    if optimality:
        chosen = [max(optimality, key=lambda k: values[k])]
        while len(chosen) < optimality_limit:
            remaining = [k for k in optimality if k not in chosen]
            if not remaining:
                break
            selected = np.asarray([cuts[k].beta[columns] for k in chosen], float)
            candidate = max(remaining, key=lambda k: (
                float(np.min(np.linalg.norm(selected - cuts[k].beta[columns], axis=1))),
                float(values[k]),
                -k))
            chosen.append(candidate)
    selected_feasibility = sorted(feasibility, key=lambda k: -values[k])[:feasibility_limit]
    return chosen, selected_feasibility


@dataclass(frozen=True)
class SearchStrategies:
    """Explicit, dependency-free seams used by one block-search run."""

    constraint_policy: ConstraintPolicy = legacy_square_policy
    cut_selector: CutSelector = recent_tight_selector
    sampler: SamplerBackend = anneal


def strategies_for_options(options, sampler=None):
    policy = cover_policy if getattr(options, "sa_constraint_policy", "legacy_square") == "cover" else legacy_square_policy
    selector = coverage_selector if getattr(options, "sa_cut_selector", "recent_tight") == "coverage" else recent_tight_selector
    return SearchStrategies(constraint_policy=policy, cut_selector=selector, sampler=sampler or anneal)

