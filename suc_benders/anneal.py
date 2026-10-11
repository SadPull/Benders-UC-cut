"""Kaiwu 1.4.1 local SA. QUBO energies are search scores, never bounds."""
import contextlib
import io
from time import perf_counter

import numpy as np


def initialize(user_id, sdk_code):
    if not user_id or not sdk_code:
        raise ValueError("Fill kaiwu credentials in config.local.json")
    try:
        # SDK diagnostics may contain license details; expose only a safe error.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import kaiwu as kw
            kw.license.init(user_id=user_id, sdk_code=sdk_code)
    except Exception:
        raise RuntimeError("Kaiwu initialization failed; check SDK 1.4.1 and local authorization") from None


def anneal(q, o, seconds, seed):
    import kaiwu as kw
    started = perf_counter()
    samples = []
    n = q.Q.shape[0]
    if seconds > 0 and n:
        scale = max(1., float(abs(q.Q).max()))
        # Kaiwu requires a dense Ising matrix; materialize only the current QUBO.
        ising, _ = kw.core.qubo_matrix_to_ising_matrix(q.Q.toarray() / scale)
        flip_scale = max(float(4 * np.abs(ising).sum(axis=1).max()), q.energy_unit / scale)
        try:
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=o.sa_temperature * flip_scale, alpha=o.sa_cooling,
                cutoff_temperature=o.sa_cutoff * q.energy_unit / scale,
                iterations_per_t=o.sa_sweeps, size_limit=o.sa_reads,
                rand_seed=int(seed), verbose=False)
            spins = np.atleast_2d(optimizer.solve(ising))
        except Exception as error:
            raise RuntimeError(f"Kaiwu sampling failed ({type(error).__name__}); check local SDK authorization") from None
        if not np.isin(spins, [-1, 1]).all() or spins.shape[1] not in (n, n + 1):
            raise RuntimeError("Kaiwu returned invalid spins")
        binary = ((1 + spins[:, :n] * spins[:, -1:]) // 2 if spins.shape[1] == n + 1
                  else (1 + spins) // 2)
        samples.extend(binary)
    energies = [float(x @ q.Q @ x + q.offset) for x in samples]
    candidates = []
    for sample in samples:
        x = q.fixed.copy(); x[q.free] = sample[:len(q.free)]
        x = x[:q.nx].astype(np.uint8)
        candidates.append(x)
    statistics = {"bits": n, "candidates": len(candidates),
                  "best_energy": min(energies, default=np.inf), "seconds": perf_counter() - started}
    if getattr(o, 'paper_sa', False):
        statistics.update(raw_bits=[''.join(str(int(v)) for v in x) for x in samples], raw_energies=energies)
    return candidates, statistics
