"""Small strategy seams for constraint encoding, cut selection and sampling."""
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .anneal import anneal, SA_BACKEND, SA_INITIALIZATION
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


def coverage_selector(cuts, values, columns, optimality_limit, feasibility_limit, *, point):
    """Greedily close the pool/workset envelope gap at binary witness points."""
    optimality = [k for k, cut in enumerate(cuts) if cut.optimality]
    feasibility = [k for k, cut in enumerate(cuts) if not cut.optimality]
    chosen = []
    if optimality and optimality_limit:
        tight = max(range(len(optimality)), key=lambda j: values[optimality[j]])
        slopes = np.asarray([cuts[k].beta[columns] for k in optimality], float)
        intercepts = np.asarray([values[k] for k in optimality]) - slopes @ point
        difference = slopes - slopes[tight]
        upper = intercepts-intercepts[tight] + np.maximum(difference, 0).sum(axis=1)
        witnesses = [j for j in np.argsort(-upper, kind='stable') if upper[j] > 1e-12][:64]
        # shortcut: 用最多 64 个割见证点及锚点/端点评估增益；全盒域保证需要精确分离。
        probes = np.unique(np.vstack([point, np.zeros(len(columns)), np.ones(len(columns)),
                                      (difference[witnesses] > 0).astype(float)]), axis=0)
        scores = intercepts[:, None] + slopes @ probes.T
        pool_envelope = scores.max(axis=0)
        selected = [tight]
        envelope = scores[tight].copy()
        while len(selected) < optimality_limit:
            gain = np.minimum(pool_envelope-envelope, np.maximum(scores-envelope, 0)).sum(axis=1)
            gain[selected] = -np.inf
            candidate = max(range(len(optimality)), key=lambda j: (gain[j], values[optimality[j]], -optimality[j]))
            if gain[candidate] <= 1e-12:
                break
            selected.append(candidate)
            envelope = np.maximum(envelope, scores[candidate])
        chosen = [optimality[j] for j in selected]
    selected_feasibility = sorted(feasibility, key=lambda k: -values[k])[:feasibility_limit]
    return chosen, selected_feasibility


def sampler_metadata(sampler):
    name = getattr(sampler, '__name__', type(sampler).__name__)
    qualified = f"{getattr(sampler, '__module__', type(sampler).__module__)}.{name}"
    return dict(sampler=name,
                sa_backend=getattr(sampler, 'sa_backend', SA_BACKEND if sampler is anneal else qualified),
                sa_initialization=getattr(sampler, 'sa_initialization', SA_INITIALIZATION if sampler is anneal else None))


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

