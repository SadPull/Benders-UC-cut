"""Strict nested block Benders/SA and Lagrangian cutting-plane search."""
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from . import llm
from .anneal import anneal
from .benders import coordinate, dispatch, master
from .model import model, partition
from .optimization import optimize
from .qubo import qubo
from .validation import repair, unpack, validate


@dataclass
class Result:
    status: str = "MAX_ITERATIONS"
    message: str = ""
    objective: float = np.inf
    lower_bound: float = -np.inf
    relative_gap: float = np.inf
    coordination_upper: float = np.inf
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    w: np.ndarray | None = None
    scenarios: list = field(default_factory=list)
    validation: dict | None = None
    history: list = field(default_factory=list)
    outer_history: list = field(default_factory=list)
    llm_calls: int = 0
    llm_history: list = field(default_factory=list)
    elapsed_seconds: float = 0.


def closed(lower, upper, tolerance, reference=None):
    reference = upper if reference is None else reference
    return (np.isfinite(lower) and np.isfinite(upper) and upper - lower >= -1e-5
            and upper - lower <= 1e-5 + tolerance * max(1, abs(reference)))


def search_block(b, seeds, o, deadline, used_calls, outer, index, rng):
    result = NS(lower=float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant),
                upper=np.inf, best=None, planes=[], seeds=[], history=[], llm_calls=0, llm_history=[])
    cuts, cache, recent = [], set(), []

    def remaining():
        return deadline - perf_counter()

    def evaluate(x, source):
        key = np.asarray(x, np.uint8).tobytes()
        if remaining() <= 0 or key in cache:
            return
        r = dispatch(b, x, remaining())
        if remaining() <= 0:
            return
        if r.feasible:
            residual = b.A @ r.point - b.rhs
            residual[b.eq] = np.abs(residual[b.eq])
            if residual.max(initial=0) > 1e-5:
                raise RuntimeError("Local dispatch validation failed")
        elif r.status not in ("INFEASIBLE", "DISCRETE_INFEASIBLE"):
            raise RuntimeError(f"Local dispatch failed: {r.status}")
        if remaining() <= 0:
            return
        cache.add(key)
        if r.cut is not None:
            cuts.append(r.cut)
        if r.feasible:
            p = r.point
            # Coordination needs the plane, not another retained dispatch vector.
            result.planes.append((float(b.original_obj @ p + b.q @ (p ** 2)), b.crossA @ p - b.crossrhs))
            if r.cost < result.upper:
                result.best, result.upper = r, r.cost
        recent.append({"source": source, "status": r.status, "cost": float(r.cost) if np.isfinite(r.cost) else None})
        del recent[:-12]

    unique = []
    for seed in seeds:
        if not any(np.array_equal(seed, x) for x in unique):
            unique.append(seed)
            evaluate(seed, "initial")
    stalled = 0
    for inner in range(1, o.max_inner + 1):
        if remaining() <= 0:
            break
        before = result.upper
        m, bound = master(b, cuts, min(o.certify_seconds, remaining()))
        if remaining() <= 0:
            break
        result.lower = max(result.lower, bound)
        sampling = {"bits": 0, "candidates": 0, "best_energy": None, "seconds": 0.}
        if not closed(result.lower, result.upper, o.tolerance):
            if result.best is None:
                raise RuntimeError("No feasible block seed")
            seed_pool = [result.best.point[b.binary], *unique]
            m.lb[-1] = max(m.lb[-1], result.lower - b.constant)
            q = qubo(m, o, seed_pool, result.upper - b.constant)
            if remaining() <= 0:
                break
            proposals, sampling = anneal(q, o, remaining(), rng.integers(0, 2 ** 31))
            del q
            for x in proposals:
                evaluate(x, "SA")
        if remaining() <= 0:
            break
        stalled = stalled + 1 if np.isfinite(before) and before - result.upper <= 5e-4 * max(1, abs(before)) else 0
        if (o.llm_enabled and result.best is not None and stalled >= o.llm_patience
                and used_calls + result.llm_calls < o.llm_max_calls
                and not closed(result.lower, result.upper, o.tolerance)):
            proposals, info = llm.request(b, result.best, result.lower, o, remaining(),
                                          {"cuts": cuts[-8:], "recent": recent})
            result.llm_calls += 1; stalled = 0
            info.update(outer=outer, block=index, inner=inner)
            result.llm_history.append(info)
            for x in proposals:
                evaluate(x, "LLM")
        if remaining() <= 0:
            break
        _, bound = master(b, cuts, min(o.certify_seconds, remaining()))
        if remaining() <= 0:
            break
        result.lower = max(result.lower, bound)
        result.history.append({"outer": outer, "block": index, "inner": inner, "lower": result.lower,
                               "upper": result.upper, "optimality_cuts": sum(c.optimality for c in cuts),
                               "feasibility_cuts": sum(not c.optimality for c in cuts), "sampling": sampling})
        if closed(result.lower, result.upper, o.tolerance):
            break
    if result.best is not None:
        result.seeds = [result.best.point[b.binary].astype(np.uint8)]
    return result


def solve(d, scenarios, o):
    started = perf_counter(); deadline = started + o.time_limit
    result = Result()
    incumbent = None
    stage = "INITIALIZATION"
    rng = np.random.default_rng(o.seed)

    def remaining():
        return deadline - perf_counter()

    def accept_full(x):
        nonlocal incumbent
        if x is None or remaining() <= 0 or (incumbent is not None and np.array_equal(x, incumbent[full.binary])):
            return
        r = dispatch(full, x, remaining())
        if not r.feasible or remaining() <= 0:
            return
        z = unpack(m, r.point)
        report = validate(d, scenarios, z)
        if remaining() > 0 and z.objective < result.objective:
            incumbent = r.point
            for name in ("objective", "u", "v", "w", "scenarios"):
                setattr(result, name, getattr(z, name))
            result.validation = report

    try:
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        relaxation = optimize(m, remaining())
        if remaining() <= 0:
            raise TimeoutError
        if relaxation.status != "OPTIMAL":
            raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
        result.lower_bound = relaxation.lower - 1e-5
        prices = relaxation.prices
        for u in (np.floor(relaxation.x[m.ix.u] + .5), np.ones((d.n_units, d.horizon))):
            accept_full(repair(d, scenarios, u))
        if incumbent is None:
            raise RuntimeError("No validated initial schedule")
        blocks, cross = partition(m, o.block_hours, prices)
        planes, previous = [[] for _ in blocks], [[] for _ in blocks]
        for outer in range(1, o.max_outer + 1):
            if remaining() <= 0:
                raise TimeoutError
            blocks, _ = partition(m, o.block_hours, prices)
            bounds = np.asarray([np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant for b in blocks])
            complete = True
            for j, b in enumerate(blocks):
                if remaining() <= 0:
                    raise TimeoutError
                stage = f"BLOCK_{outer}_{j + 1}"
                seeds = [incumbent[b.cols[b.binary]], *previous[j]]
                search = search_block(b, seeds, o, deadline, result.llm_calls, outer, j + 1, rng)
                result.llm_calls += search.llm_calls
                result.llm_history.extend(search.llm_history)
                result.history.extend(search.history)
                bounds[j] = max(bounds[j], search.lower)
                result.lower_bound = max(result.lower_bound, float(bounds.sum()))
                previous[j] = search.seeds
                planes[j].extend(search.planes)
                if search.best is not None and remaining() > 0:
                    proposal = result.u.copy(); proposal[:, b.times] = search.best.point[b.u]
                    accept_full(repair(d, scenarios, proposal))
                if o.verbose:
                    print(f"outer={outer} block={j + 1}/{len(blocks)} local=[{search.lower:.6f}, {search.upper:.6f}] "
                          f"full={result.objective:.6f} LB={result.lower_bound:.6f}")
                if remaining() <= 0:
                    raise TimeoutError
                if not closed(search.lower, search.upper, o.tolerance):
                    result.status = "INNER_NOT_CONVERGED"; complete = False
                    break
            if not complete:
                break
            stage = "COORDINATION"
            values, upper = coordinate(planes, m.eq[cross], result.objective, remaining())
            if remaining() <= 0:
                raise TimeoutError
            result.coordination_upper = upper
            result.outer_history.append({"outer": outer, "lower": result.lower_bound, "coordination_upper": upper})
            if closed(result.lower_bound, upper, o.tolerance, result.lower_bound):
                result.status = "DUAL_CONVERGED"
                break
            prices = np.zeros(len(m.rhs)); prices[cross] = values
    except Exception as error:
        result.status = "TIME_LIMIT" if remaining() <= 0 or isinstance(error, TimeoutError) else f"{stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if result.lower_bound > result.objective + 1e-4:
            raise RuntimeError("Lower bound exceeds validated cost")
        result.relative_gap = max(0., result.objective - result.lower_bound) / max(1., abs(result.objective))
    result.elapsed_seconds = perf_counter() - started
    if o.verbose:
        print(f"{result.status}: cost={result.objective:.6f}, lower={result.lower_bound:.6f}, "
              f"gap={100 * result.relative_gap:.4f}%, elapsed={result.elapsed_seconds:.2f}s")
    return result
