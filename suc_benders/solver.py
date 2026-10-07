"""Nested block search with archived SA/repair and whole-day advisor policies."""
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from . import llm
from .advisor_feedback import call_due, summarize
from .anneal import anneal
from .benders import coordinate, dispatch, master
from .context import CandidateEvaluation, LLMContextInput, build_llm_context
from .model import model, partition
from .optimization import optimize
from .qubo import qubo
from .validation import capacity, commitment, commitment_hash, pack, repair, unpack, validate


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
    diagnostics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.


def closed(lower, upper, tolerance, reference=None):
    reference = upper if reference is None else reference
    return (np.isfinite(lower) and np.isfinite(upper) and upper - lower >= -1e-5
            and upper - lower <= 1e-5 + tolerance * max(1, abs(reference)))


def discrete_feasible(b, x):
    hard = b.A[:, b.continuous].getnnz(axis=1) == 0
    residual = b.A[hard][:, b.binary] @ x - b.rhs[hard]
    residual[b.eq[hard]] = np.abs(residual[b.eq[hard]])
    return bool(np.isin(x, [0, 1]).all() and residual.max(initial=0) <= 1e-7)


class WholeDay:
    """Shared incumbent, measured feedback and deadline across all time blocks."""

    def __init__(self, m, full, o, deadline, result, started):
        self.m, self.full, self.o, self.deadline = m, full, o, deadline
        self.d, self.scenarios, self.result, self.started = m.d, m.scenarios, result, started
        self.best, self.dispatches = None, []
        self.cache, self.advisor_pool = {}, []
        self.stalled, self.feedback = 0, None
        self.diagnostics = result.diagnostics
        self.diagnostics.update(sa_calls=0, evaluations=[], proposals=[], sampling=[], iterations=[],
                                local_evaluations=[], llm_calls=[], timings={name: 0. for name in (
                                    "initialization", "relaxation", "qubo", "annealing", "repair",
                                    "local_dispatch", "subproblems", "validation", "context", "llm", "coordination")})
        self.timings = self.diagnostics["timings"]

    def remaining(self):
        return self.deadline - perf_counter()

    def prepare(self, raw, source, record):
        """Repair a whole-day u proposal; the caller then evaluates its block projection."""
        record.update(origin=source, raw_u=np.asarray(raw).tolist())
        self.diagnostics["proposals"].append(record)
        if self.remaining() <= 0:
            record["outcome"] = "deadline"
            return None
        raw = np.asarray(raw, float)
        if raw.shape != (self.d.n_units, self.d.horizon) or not np.isfinite(raw).all():
            record["outcome"] = "invalid_matrix"
            return None
        raw_x = pack(self.d, raw)
        record["raw_binary"] = bool(np.isin(raw, [0, 1]).all())
        z = commitment(self.d, raw_x)
        required = (1 + self.d.reserve_fraction) * np.max([s.load for s in self.scenarios], axis=0)
        record["raw_discrete_feasible"] = bool(discrete_feasible(self.full, raw_x)
            and (capacity(self.d, z.u, z.v, z.w).sum(axis=0) >= required - 1e-6).all())
        phase = perf_counter()
        x = repair(self.d, self.scenarios, raw)
        self.timings["repair"] += perf_counter() - phase
        if self.remaining() <= 0:
            record["outcome"] = "deadline"
            return None
        if x is None:
            record["outcome"] = "repair_failed"
            return None
        z = commitment(self.d, x)
        record.update(commitment_id=commitment_hash(z), repaired_u=z.u.tolist(),
                      hamming_u=int(np.count_nonzero(z.u != raw)), repaired_discrete_feasible=True,
                      duplicate=x.astype(np.uint8).tobytes() in self.cache, outcome="repaired")
        return x

    def accept(self, x, source, record=None):
        if x is None or self.remaining() <= 0:
            if record is not None:
                record["outcome"] = "deadline"
            return False
        key = x.astype(np.uint8).tobytes()
        if key in self.cache:
            if record is not None:
                record.update(outcome="duplicate", duplicate=True, cache_hit=True)
            return self.cache[key]["feasible"]
        phase = perf_counter()
        r = dispatch(self.full, x, self.remaining())
        self.timings["subproblems"] += perf_counter() - phase
        if self.remaining() <= 0 or r.status == "TIME_LIMIT":
            if record is not None:
                record["outcome"] = "dispatch_timeout"
            return False
        if r.status not in ("OPTIMAL", "INFEASIBLE", "DISCRETE_INFEASIBLE"):
            raise RuntimeError(f"Full dispatch failed: {r.status}")
        before = self.result.objective
        z = commitment(self.d, x)
        fixed = float(np.sum(self.d.cost_c[:, None] * z.u + self.d.startup_cost[:, None] * z.v
                             + self.d.shutdown_cost[:, None] * z.w))
        report, scenario_costs = None, {}
        if r.feasible:
            z = unpack(self.m, r.point)
            phase = perf_counter()
            report = validate(self.d, self.scenarios, z)
            self.timings["validation"] += perf_counter() - phase
            scenario_costs = {s.name: s.objective for s in z.scenarios}
        if self.remaining() <= 0:
            if record is not None:
                record["outcome"] = "validation_deadline"
            return False
        improved = r.feasible and z.objective < before
        entry = dict(source=source, feasible=r.feasible, objective=z.objective if r.feasible else None,
                     fixed_cost=fixed, expected_fuel_cost=z.objective-fixed if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=commitment_hash(z),
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        self.cache[key] = entry
        self.diagnostics["evaluations"].append(entry)
        if record is not None:
            record.update(outcome="accepted" if r.feasible else "infeasible", feasible=r.feasible,
                          continuous_dispatch_completed=True, validation_completed=r.feasible)
        if improved:
            self.best = r
            for name in ("objective", "u", "v", "w", "scenarios"):
                setattr(self.result, name, getattr(z, name))
            self.result.validation = report
            # Joint QP prices contain scenario probability and use A*x <= rhs signs.
            labels = np.asarray(self.full.labels)
            balance = r.prices[labels == "balance"].reshape((-1, len(self.scenarios)))
            reserve = r.prices[labels == "reserve"].reshape((-1, len(self.scenarios)))
            self.dispatches = [NS(**vars(z.scenarios[k]),
                balance_dual=-balance[:, k] / s.probability,
                reserve_dual=reserve[:, k] / s.probability) for k, s in enumerate(self.scenarios)]
        if source == "llm" and r.feasible:
            self.advisor_pool.append((r.cost, x.copy()))
            self.advisor_pool.sort(key=lambda item: item[0])
            del self.advisor_pool[16:]
        return r.feasible

    def after_batch(self, before, b, evaluate, outer, index, inner, allow_advice):
        due, self.stalled = call_due("stagnation_burst", before, self.result.objective,
                                     self.stalled, self.result.llm_calls, self.o.llm_patience)
        if not (allow_advice and due and self.o.llm_enabled):
            return
        for _ in range(3):
            if self.remaining() <= 0 or self.result.llm_calls >= self.o.llm_max_calls:
                break
            phase = perf_counter()
            measurements = self.diagnostics["evaluations"]
            evaluations = [CandidateEvaluation(r["source"], r["objective"] if r["feasible"] else np.inf,
                r["fixed_cost"], r["expected_fuel_cost"] if r["feasible"] else np.inf,
                r["scenario_costs"], r["feasible"], 0, 0, None) for r in measurements]
            context = build_llm_context(self.d, self.scenarios, LLMContextInput(
                self.result, self.dispatches, evaluations[-16:], evaluations,
                [NS(**row) for row in self.diagnostics["iterations"]]))
            context["previous_call_feedback"] = self.feedback
            baseline_u = self.result.u.copy()
            baseline = next(r for r in reversed(measurements) if r["commitment_hash"] == commitment_hash(self.result))
            costs = dict(total=baseline["objective"], fixed=baseline["fixed_cost"], expected_fuel=baseline["expected_fuel_cost"])
            prior_ids = {r["commitment_hash"] for r in measurements}
            pool = sorted(((key, row) for key, row in self.cache.items() if row["feasible"]),
                          key=lambda item: item[1]["objective"])[:16]
            bases = [commitment(self.d, np.frombuffer(key, dtype=np.uint8)) for key, _ in pool]
            self.timings["context"] += perf_counter() - phase
            if self.remaining() <= 0:
                break
            self.result.llm_calls += 1
            self.stalled = 0
            phase = perf_counter()
            proposals, info = llm.request(self.d, self.scenarios, self.result, self.o, self.remaining(),
                                         context, bases, self.result.llm_calls)
            self.timings["llm"] += perf_counter() - phase
            info.update(outer=outer, block=index, inner=inner, call_id=f"llm:{self.result.llm_calls}")
            self.result.llm_history.append(info)
            self.diagnostics["llm_calls"].append(info)
            if self.remaining() > 0:
                for raw, record in proposals:
                    x = self.prepare(raw, "llm", record)
                    if x is None or self.remaining() <= 0:
                        continue
                    local = evaluate(x[b.cols[b.binary]], "llm")
                    record.update(local_dispatch_completed=local is not None,
                                  local_feasible=local.feasible if local is not None else None)
                    self.accept(x, "llm", record)
            else:
                for record in info.get("proposals", []):
                    record["outcome"] = "deadline"
            info["accepted"] = sum(p.get("outcome") == "accepted" for p in info.get("proposals", []))
            self.feedback = summarize(NS(diagnostics=info, failed=info["status"] != "OK",
                                         message=info.get("message", "")),
                                      baseline_u, costs, measurements, prior_ids)
            info["feedback"] = self.feedback


def search_block(b, seeds, o, deadline, outer, index, whole):
    result = NS(lower=float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant),
                upper=np.inf, best=None, planes=[], seeds=[], history=[])
    cuts, cache = [], {}

    def remaining():
        return deadline - perf_counter()

    def evaluate(x, source):
        key = np.asarray(x, np.uint8).tobytes()
        if remaining() <= 0:
            return
        if key in cache:
            return cache[key]
        phase = perf_counter()
        r = dispatch(b, x, remaining())
        whole.timings["local_dispatch"] += perf_counter() - phase
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
        cache[key] = r
        if r.cut is not None:
            cuts.append(r.cut)
        if r.feasible:
            p = r.point
            result.planes.append((float(b.original_obj @ p + b.q @ (p ** 2)), b.crossA @ p - b.crossrhs))
            if r.cost < result.upper:
                result.best, result.upper = r, r.cost
        whole.diagnostics["local_evaluations"].append(dict(outer=outer, block=index, source=source,
            feasible=r.feasible, cost=r.cost if r.feasible else None))
        return r

    for seed in seeds:
        evaluate(seed, "initial")
    for inner in range(1, o.max_inner + 1):
        if remaining() <= 0:
            break
        phase = perf_counter()
        m, bound = master(b, cuts, remaining())
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            break
        result.lower = max(result.lower, bound)
        sampling = {"bits": 0, "candidates": 0, "best_energy": None, "seconds": 0.}
        before_count, before_opt = len(whole.diagnostics["evaluations"]), sum(c.optimality for c in cuts)
        before_feas = len(cuts) - before_opt
        before_calls = whole.result.llm_calls
        if not closed(result.lower, result.upper, o.tolerance):
            if result.best is None:
                raise RuntimeError("No feasible block seed")
            before = whole.result.objective
            baseline_u = whole.result.u.copy()
            phase = perf_counter()
            m.lb[-1] = max(m.lb[-1], result.lower - b.constant)
            q = qubo(m, o, result.upper - b.constant)
            whole.timings["qubo"] += perf_counter() - phase
            if remaining() <= 0:
                break
            phase = perf_counter()
            whole.diagnostics["sa_calls"] += 1
            proposals, sampling = anneal(q, o, remaining(), o.seed)
            whole.timings["annealing"] += perf_counter() - phase
            sampling.update(outer=outer, block=index, inner=inner, seed=o.seed,
                            raw_candidates=len(proposals), raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals),
                            unique_uvw=len({x.astype(np.uint8).tobytes() for x in proposals}))
            whole.diagnostics["sampling"].append(sampling)
            del q
            if remaining() <= 0:
                break
            # All samples use the same incumbent snapshot, before this batch's improvements.
            for k, x in enumerate(proposals):
                if remaining() <= 0:
                    break
                local = np.zeros(len(b.obj)); local[b.binary] = x
                raw = baseline_u.copy(); raw[:, b.times] = local[b.u]
                record = {"proposal_id": f"sa:{whole.diagnostics['sa_calls']}:{k}", "outer": outer, "block": index}
                repaired = whole.prepare(raw, "sa", record)
                if repaired is not None and remaining() > 0:
                    local = evaluate(repaired[b.cols[b.binary]], "sa")
                    record.update(local_dispatch_completed=local is not None,
                                  local_feasible=local.feasible if local is not None else None)
                    whole.accept(repaired, "sa", record)
            if remaining() > 0:
                whole.after_batch(before, b, evaluate, outer, index, inner,
                                  not closed(result.lower, result.upper, o.tolerance))
        if remaining() <= 0:
            break
        phase = perf_counter()
        _, bound = master(b, cuts, remaining())
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            break
        result.lower = max(result.lower, bound)
        optimality = sum(c.optimality for c in cuts)
        result.history.append(dict(outer=outer, block=index, inner=inner, lower=result.lower,
            upper=result.upper, optimality_cuts=optimality, feasibility_cuts=len(cuts)-optimality, sampling=sampling))
        whole.diagnostics["iterations"].append(dict(iteration=len(whole.diagnostics["iterations"])+1,
            incumbent=whole.result.objective, optimality_cuts_added=optimality-before_opt,
            feasibility_cuts_added=len(cuts)-optimality-before_feas,
            new_candidates=len(whole.diagnostics["evaluations"])-before_count,
            llm_failed=any(row["status"] != "OK" for row in whole.result.llm_history[before_calls:])))
        if closed(result.lower, result.upper, o.tolerance):
            break
    pool = sorted((r for r in cache.values() if r.feasible), key=lambda r: r.cost)[:16]
    result.seeds = [r.point[b.binary].astype(np.uint8) for r in pool]
    return result


def solve(d, scenarios, o):
    started = perf_counter(); deadline = started + o.time_limit
    result, whole = Result(), None
    stage = "INITIALIZATION"

    def remaining():
        return deadline - perf_counter()

    try:
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        whole = WholeDay(m, full, o, deadline, result, started)
        phase = perf_counter()
        relaxation = optimize(m, remaining())
        whole.timings["initialization"] += perf_counter() - phase
        if remaining() <= 0:
            raise TimeoutError
        if relaxation.status != "OPTIMAL":
            raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
        result.lower_bound = relaxation.objval - 1e-5
        prices = relaxation.prices
        for u in (np.floor(relaxation.x[m.ix.u] + .5), np.ones((d.n_units, d.horizon))):
            record = {"proposal_id": f"initial:{len(whole.diagnostics['proposals'])}"}
            x = whole.prepare(u, "initial", record)
            whole.accept(x, "initial", record)
        if whole.best is None:
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
                seeds = [whole.best.point[b.cols[b.binary]], *previous[j],
                         *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
                search = search_block(b, seeds, o, deadline, outer, j + 1, whole)
                result.history.extend(search.history)
                bounds[j] = max(bounds[j], search.lower)
                result.lower_bound = max(result.lower_bound, float(bounds.sum()))
                previous[j] = search.seeds
                planes[j].extend(search.planes)
                if search.best is not None and remaining() > 0:
                    raw = result.u.copy(); raw[:, b.times] = search.best.point[b.u]
                    record = {"proposal_id": f"pool:{outer}:{j+1}"}
                    x = whole.prepare(raw, "pool", record)
                    whole.accept(x, "pool", record)
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
            phase = perf_counter()
            values, upper = coordinate(planes, m.eq[cross], result.objective, remaining())
            whole.timings["coordination"] += perf_counter() - phase
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
    if whole is not None:
        whole.timings["other"] = max(0., result.elapsed_seconds - sum(whole.timings.values()))
        result.diagnostics.update(evaluation_count=len(whole.cache), llm_call_count=result.llm_calls,
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit))
    if o.verbose:
        print(f"{result.status}: cost={result.objective:.6f}, lower={result.lower_bound:.6f}, "
              f"gap={100 * result.relative_gap:.4f}%, elapsed={result.elapsed_seconds:.2f}s")
    return result
