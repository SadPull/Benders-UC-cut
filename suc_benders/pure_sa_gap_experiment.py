"""Prespecified pure-SA search and separate integer certification protocol."""
from __future__ import annotations

from .master import QuantumConfig
from .sa_scale_experiment import SEEDS
from .solver import BendersConfig


HOURS = (1, 2, 3, 5)
VARIANTS = ("sa_strict", "sa_flow", "sa_compact", "sa_full")
PHASE_SECONDS = dict(reference=720, sanity=360, diagnostic=2400, screening=1680,
                     ablation=1680, comparison=1860, confirmation=5400, audit=300)
TOTAL_SECONDS = sum(PHASE_SECONDS.values())


def definitions():
    runs = []
    def add(stage, case, hours, method, seed, search, audit):
        runs.append(dict(id=f"{stage}-{case}-h{hours or 'selected'}-{method}-{seed}",
                         stage=stage, case=case, hours=hours, method=method, seed=seed,
                         seconds=search, audit_seconds=audit))
    for seed in SEEDS:
        add("sanity", "n3-det", 1, "sa_full", seed, 60, 20)
    add("sanity", "n10-det", 1, "sa_full", 7, 90, 30)
    for hours in HOURS:
        add("diagnostic", "n32-stoch", hours, "native_diagnostic", 7, 600, 0)
    for hours in HOURS:
        add("screening", "n32-stoch", hours, "sa_full", 7, 300, 120)
    for variant in VARIANTS:
        add("ablation", "n32-stoch", None, variant, 7, 300, 120)
    for seed in SEEDS:
        add("comparison", "n32-stoch", None, "selected", seed, 300, 120)
    add("comparison", "n32-stoch", None, "native_diagnostic", 7, 600, 0)
    for seed in SEEDS:
        add("confirmation", "n32-stoch", None, "selected", seed, 1200, 600)
    return runs


def configuration(definition, seconds=None):
    method = definition["method"]
    if method == "selected":
        raise ValueError("resolve selected variant before starting a search worker")
    native = method == "native_diagnostic"
    return BendersConfig(max_iterations=40, max_inner_iterations=200 if method == "sa_strict" else 8,
        block_hours=definition["hours"], convergence_tolerance=1e-4,
        quantum_candidates=not native, native_fallback_patience=1 if native else None,
        defer_full_recovery=True, gap_experiment_variant=method, sa_seed_per_call=True,
        quantum=QuantumConfig(kaiwu_seed=definition["seed"]),
        time_limit_seconds=definition["seconds"] if seconds is None else seconds)
