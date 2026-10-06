from __future__ import annotations

import numpy as np

from .master import QuantumConfig


def solve_qubo(
    matrix: np.ndarray, config: QuantumConfig, *, energy_unit: float = 1.0
) -> list[np.ndarray]:
    """Solve a QUBO with the only implemented backend: Kaiwu local SA."""

    if config.backend != "kaiwu-local":
        raise RuntimeError(f"quantum backend {config.backend!r} is not configured; use kaiwu-local")

    from .common import get_kaiwu

    kw = get_kaiwu()
    if not np.isfinite(energy_unit) or energy_unit <= 0:
        raise ValueError("energy_unit must be finite and positive")
    matrix = np.asarray(matrix, dtype=float)
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 1.0)
    normalized = matrix / scale
    ising, _ = kw.core.qubo_matrix_to_ising_matrix(normalized)
    # For the SDK's symmetric Ising matrix, a spin flip changes energy by
    # at most four times its absolute row sum. Cool to objective resolution,
    # not to a fixed fraction of the largest constraint coefficient.
    flip_scale = max(float(4.0 * np.max(np.sum(np.abs(ising), axis=1))), energy_unit / scale)
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=config.initial_temperature * flip_scale,
        alpha=config.alpha,
        cutoff_temperature=config.cutoff_temperature * energy_unit / scale,
        iterations_per_t=config.iterations_per_temperature,
        size_limit=config.candidate_pool_size,
        rand_seed=config.kaiwu_seed,
        verbose=False,
    )
    spins = np.asarray(optimizer.solve(ising))
    if spins.ndim == 1:
        spins = spins[None, :]

    # Kaiwu appends an auxiliary spin for linear terms; decode all spins in
    # the documented binary convention used by the prior verified project.
    n_variables = matrix.shape[0]
    binaries: list[np.ndarray] = []
    for row in spins:
        row = np.asarray(row, dtype=int).reshape(-1)
        if row.size == n_variables + 1:
            binary = ((1 + row[:n_variables] * row[-1]) // 2).astype(int)
        else:
            binary = ((row[:n_variables] + 1) // 2).astype(int)
        binaries.append(binary)
    return binaries
