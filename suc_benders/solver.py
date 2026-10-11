"""Round-robin block search with shared whole-day validation and advice."""
from dataclasses import dataclass, field
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from . import llm
from .advisor_feedback import call_due, summarize
from .anneal import anneal
from .benders import coordinate, dispatch, master
from .certification import Certification
from .context import CandidateEvaluation, LLMContextInput, build_llm_context
from .dual_advisor import DualAdvisor
from .model import fingerprint, model, partition
from .optimization import continuous_only, optimize
from .paper import context_id, process_raw_batch, record_bound, record_cut, record_state, sampling_evidence
from .qubo import qubo
from .startup import fixed_cost
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
    certification_history: list = field(default_factory=list)
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

    def __init__(self, m, full, o, deadline, result, started, progress=None):
        if o.paper_sa and (o.bound_mode != 'relaxation' or o.llm_enabled or not o.sa_enabled):
            raise ValueError('Paper SA cannot create an integer certifier or LLM advisor')
        self.m, self.full, self.o, self.deadline = m, full, o, deadline
        self.d, self.scenarios, self.result, self.started = m.d, m.scenarios, result, started
        self.best, self.dispatches = None, []
        self.cache, self.advisor_pool = {}, []
        self.repair_cache = {}
        self.stalled, self.feedback = 0, None
        self.diagnostics = result.diagnostics
        self.diagnostics.update(sa_calls=0, evaluations=[], proposals=[], sampling=[], iterations=[],
                                local_evaluations=[], llm_calls=[], blocks=[], rounds=[],
                                complete_rounds=0, coordination_count=0, timings={name: 0. for name in (
                                    "initialization", "relaxation", "qubo", "annealing", "repair",
                                    "local_dispatch", "subproblems", "validation", "context", "llm", "coordination")})
        self.timings = self.diagnostics["timings"]
        self.certifier = Certification(self) if o.bound_mode == "integer_benders" else None
        self.diagnostics["bound_mode"] = o.bound_mode
        self.diagnostics["model_fingerprint"] = fingerprint(m)
        self.resume_cursor = dict(outer=1, round=1)
        self.advice = DualAdvisor(self) if self.certifier is not None else None
        self.progress = progress

    def remaining(self):
        return self.deadline - perf_counter()

    def emit(self, event):
        if self.progress is not None:
            self.progress(self, event)

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
            and (getattr(self.d, "model_profile", "") == "azema_sms"
                 or (capacity(self.d, z.u, z.v, z.w).sum(axis=0) >= required - 1e-6).all()))
        phase = perf_counter()
        repair_key = raw.astype('<f8').tobytes()
        if (self.o.rts24_improvements or self.o.paper_sa) and repair_key in self.repair_cache:
            cached = self.repair_cache[repair_key]
            x = None if cached is None else cached.copy()
            record['repair_cache_hit'] = True
        else:
            x = repair(self.d, self.scenarios, raw)
            if self.o.rts24_improvements or self.o.paper_sa:
                self.repair_cache[repair_key] = None if x is None else x.copy()
            record['repair_cache_hit'] = False
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
                record.update(outcome="duplicate", duplicate=True, cache_hit=True, incumbent_decrease=0.)
            return self.cache[key]["feasible"]
        phase = perf_counter()
        r = (dispatch(self.full, x, self.remaining(), certified=True) if self.certifier is not None or self.o.paper_sa
             else dispatch(self.full, x, self.remaining()))
        self.timings["subproblems"] += perf_counter() - phase
        if self.remaining() <= 0 or r.status == "TIME_LIMIT":
            if record is not None:
                record["outcome"] = "dispatch_timeout"
            return False
        if r.status not in ("OPTIMAL", "INFEASIBLE", "DISCRETE_INFEASIBLE"):
            raise RuntimeError(f"Full dispatch failed: {r.status}")
        before = self.result.objective
        z = commitment(self.d, x)
        fixed = fixed_cost(self.d, z.u, z.w)
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
        if self.certifier is not None:
            self.certifier.observe(x, r, source)
        improved = r.feasible and z.objective < before
        entry = dict(source=source, feasible=r.feasible, objective=z.objective if r.feasible else None,
                     fixed_cost=fixed, expected_fuel_cost=z.objective-fixed if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=commitment_hash(z),
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        if getattr(self.d, "model_profile", "") == "azema_sms" and r.feasible:
            entry.update(expected_fuel_cost=report["expected_fuel_cost"],
                         expected_penalty_cost=report["expected_penalty_cost"],
                         expected_recourse_cost=z.objective-fixed,
                         expected_shed_mwh=report["expected_shed_mwh"],
                         expected_curtail_mwh=report["expected_curtail_mwh"])
        if record is not None:
            entry["proposal_id"] = record.get("proposal_id")
            record["incumbent_decrease"] = float(before-z.objective) if improved else 0.
        self.cache[key] = entry
        if self.certifier is not None and r.feasible:
            entry["hourly_recourse_costs"] = r.hourly_recourse.tolist()
        if self.certifier is not None or self.o.paper_sa:
            entry["u"] = z.u.tolist()
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
            reserve = (r.prices[labels == "reserve"].reshape((-1, len(self.scenarios)))
                       if (labels == "reserve").any() else np.zeros_like(balance))
            self.dispatches = [NS(**vars(z.scenarios[k]),
                balance_dual=-balance[:, k] / s.probability,
                reserve_dual=reserve[:, k] / s.probability) for k, s in enumerate(self.scenarios)]
        if source in ("llm", "llm_bound") and r.feasible:
            self.advisor_pool.append((r.cost, x.copy()))
            self.advisor_pool.sort(key=lambda item: item[0])
            del self.advisor_pool[16:]
        self.emit("evaluation")
        return r.feasible

    def check_repricing(self, prices, outer):
        whole, o, m = self, self.o, self.m
        remaining = self.remaining
        # Check every next-price block before accepting a new sampling cycle.
        next_blocks, _ = partition(m, o.block_hours, prices)
        checks = []
        for j, b in enumerate(next_blocks):
            if remaining() <= 0:
                raise TimeoutError
            x = whole.best.point[b.cols[b.binary]]
            local = dispatch(b, x, remaining())
            if not local.feasible:
                raise RuntimeError('Repriced incumbent block is infeasible')
            encoded, lb = master(b, [local.cut], remaining())
            encoded.lb[-1] = max(encoded.lb[-1], lb-b.constant)
            q = qubo(encoded, o, local.cost-b.constant)
            check = dict(outer=outer, block=j+1, bits=q.Q.shape[0], nonzeros=q.Q.nnz,
                max_coefficient=float(abs(q.Q).max()), dense_working_bytes=32*(q.Q.shape[0]+1)**2)
            whole.diagnostics['pending_qubo'] = check
            whole.emit('before_sampling')
            checks.append(check)
            del q
        whole.diagnostics.setdefault('repricing_checks', []).append(dict(outer=outer,
            max_multiplier=float(abs(prices).max(initial=0)), blocks=checks))
        whole.emit('repricing')

    def restore(self, state):
        """Recover validated global evidence; block caches and multipliers are rebuilt."""
        if self.certifier is None:
            raise ValueError("Resume requires integer_benders mode")
        saved = deepcopy(state)
        diagnostic = saved["diagnostics"]
        signature = diagnostic.get("model_fingerprint")
        if signature is None:
            if hasattr(self.d, "cold_startup_cost") or getattr(self.d, "model_profile", "") == "azema_sms":
                raise ValueError("This model profile requires a resume fingerprint")
        elif signature != self.diagnostics["model_fingerprint"]:
            raise ValueError("Recovered model fingerprint does not match the current inputs")
        if np.asarray(saved["u"]).shape != (self.d.n_units, self.d.horizon):
            raise ValueError("Recovered commitment dimensions do not match the current case")
        for row in diagnostic["evaluations"]:
            if "hourly_fuel_costs" in row and "hourly_recourse_costs" not in row:
                row["hourly_recourse_costs"] = row.pop("hourly_fuel_costs")
        for row in diagnostic["bound_probes"]:
            if "hourly_actual_fuel" in row and "hourly_actual_recourse" not in row:
                row["hourly_actual_recourse"] = row.pop("hourly_actual_fuel")
        self.certifier.restore_cuts(diagnostic)
        self.result.certification_history = saved["certification_history"]
        self.result.llm_history, self.result.llm_calls = saved["llm_history"], saved["llm_calls"]
        for name in ("evaluations", "proposals", "bound_probes"):
            self.diagnostics[name] = diagnostic[name]
        self.diagnostics["llm_calls"] = self.result.llm_history
        self.diagnostics["llm_background_seconds"] = sum(row.get("background_seconds", 0.) for row in self.result.llm_history)
        self.resume_cursor = saved["resume_cursor"]
        for name in ("sa_calls", "complete_rounds", "coordination_count"):
            self.diagnostics[name] = diagnostic[name]
        self.timings["prior_execution"] = saved["charged_elapsed_seconds"]
        self.diagnostics["resume"] = saved["resume"]
        x = pack(self.d, np.asarray(saved["u"]))
        phase = perf_counter()
        checked = dispatch(self.full, x, self.remaining(), certified=True)
        self.timings["subproblems"] += perf_counter()-phase
        if not checked.feasible:
            raise RuntimeError("Recovered incumbent failed continuous dispatch")
        point = unpack(self.m, checked.point)
        report = validate(self.d, self.scenarios, point)
        if abs(point.objective-saved["objective"]) > 1e-4:
            raise RuntimeError("Recovered incumbent cost does not match saved evidence")
        self.best = checked
        for name in ("objective", "u", "v", "w", "scenarios"):
            setattr(self.result, name, getattr(point, name))
        self.result.validation = report
        labels = np.asarray(self.full.labels)
        balance = checked.prices[labels == "balance"].reshape((-1, len(self.scenarios)))
        reserve = (checked.prices[labels == "reserve"].reshape((-1, len(self.scenarios)))
                   if (labels == "reserve").any() else np.zeros_like(balance))
        self.dispatches = [NS(**vars(point.scenarios[k]), balance_dual=-balance[:, k]/s.probability,
                              reserve_dual=reserve[:, k]/s.probability)
                           for k, s in enumerate(self.scenarios)]
        self.certifier.update_bound(saved["lower_bound"])
        self.certifier.observe(x, checked, "resume_validation")
        evaluated = {row["commitment_hash"]: row for row in self.diagnostics["evaluations"]}
        known = [*self.diagnostics["bound_probes"],
                 *(dict(raw_u=row["u"]) for row in evaluated.values() if "u" in row)]
        history = self.result.certification_history
        processed_before = history[-2]["elapsed_seconds"] if len(history) > 1 else -np.inf
        for proposal in known:
            x = pack(self.d, np.asarray(proposal["raw_u"]))
            row = evaluated[commitment_hash(commitment(self.d, x))]
            key = x.astype(np.uint8).tobytes()
            if key in self.cache:
                continue
            self.cache[key] = row
            if row["feasible"] and row["elapsed_seconds"] <= processed_before:
                for part in self.certifier.hourly.parts:
                    part.cache.add(x[part.indices].astype(np.uint8).tobytes())
            if row["feasible"] and row["source"] in ("llm", "llm_bound"):
                self.advisor_pool.append((row["objective"], x.copy()))
        self.advisor_pool.sort(key=lambda item: item[0])
        del self.advisor_pool[16:]
        self.advice.restore(diagnostic)

    def upper_context(self):
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
        if "expected_penalty_cost" in baseline:
            costs["expected_penalty"] = baseline["expected_penalty_cost"]
        prior_ids = {r["commitment_hash"] for r in measurements}
        pool = sorted(((key, row) for key, row in self.cache.items() if row["feasible"]),
                      key=lambda item: item[1]["objective"])[:16]
        bases = [commitment(self.d, np.frombuffer(key, dtype=np.uint8)) for key, _ in pool]
        self.timings["context"] += perf_counter() - phase
        return context, bases, (baseline_u, costs, prior_ids)

    def after_round(self, before, searches, outer, inner, allow_advice):
        due, self.stalled = call_due("stagnation_burst", before, self.result.objective,
                                     self.stalled, self.result.llm_calls, self.o.llm_patience)
        if not (allow_advice and due and self.o.llm_enabled):
            return
        if self.remaining() <= 0 or (self.o.llm_max_calls is not None
                                   and self.result.llm_calls >= self.o.llm_max_calls):
            return
        context, bases, baseline = self.upper_context()
        if self.remaining() <= 0:
            return
        self.result.llm_calls += 1
        self.stalled = 0
        phase = perf_counter()
        proposals, info = llm.request(self.d, self.scenarios, self.result, self.o, self.remaining(),
                                     context, bases, self.result.llm_calls)
        self.timings["llm"] += perf_counter() - phase
        info.update(outer=outer, block=None, inner=inner, round=inner, stage="round_end", call_id=f"llm:{self.result.llm_calls}")
        self.result.llm_history.append(info)
        self.diagnostics["llm_calls"].append(info)
        self.consume_upper(proposals, info, searches, baseline)

    def consume_upper(self, proposals, info, searches, baseline):
        baseline_u, costs, prior_ids = baseline
        if self.remaining() > 0:
            for raw, record in proposals:
                x = self.prepare(raw, "llm", record)
                if x is None or self.remaining() <= 0:
                    continue
                if self.accept(x, "llm", record):
                    projections = []
                    for search in searches:
                        if self.remaining() <= 0:
                            break
                        b = search.b
                        local = search.evaluate(x[b.cols[b.binary]], "llm")
                        projections.append(dict(block=search.index, completed=local is not None,
                                                feasible=local.feasible if local is not None else None))
                    record["block_projections"] = projections
                    record["local_dispatch_completed"] = (len(projections) == len(searches)
                        and all(row["completed"] for row in projections))
                    record["local_feasible"] = (all(row["feasible"] for row in projections)
                        if record["local_dispatch_completed"] else None)
        else:
            for record in info.get("proposals", []):
                record["outcome"] = "deadline"
        info["accepted"] = sum(p.get("outcome") == "accepted" for p in info.get("proposals", []))
        self.feedback = summarize(NS(diagnostics=info, failed=info["status"] != "OK",
                                     message=info.get("message", "")),
                                  baseline_u, costs, self.diagnostics["evaluations"], prior_ids)
        info["feedback"] = self.feedback
        if self.advice is not None:
            self.diagnostics["upper_advice_feedback"] = self.feedback


class BlockSearch:
    """One fixed-multiplier block, retained across round-robin visits."""

    def __init__(self, b, o, outer, index, whole, planes=None):
        self.b, self.o, self.outer, self.index, self.whole = b, o, outer, index, whole
        self.context = context_id(b, outer, index)
        self.cut_keys = set()
        b.paper_sa = o.paper_sa
        if o.paper_sa:
            b.audit_cuts = True
        self.lower = float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant)
        self.upper, self.best = np.inf, None
        self.cuts, self.cache, self.history = [], {}, []
        self.planes = [] if planes is None else planes
        self.diagnostic = dict(outer=outer, block=index, hours=(b.times + 1).tolist(), visits=0,
                               sa_calls=0, first_visit_seconds=None)
        whole.diagnostics["blocks"].append(self.diagnostic)
        self.summarize()

    def summarize(self):
        self.diagnostic.update(lower=self.lower, upper=self.upper,
            relative_gap=(max(0., self.upper-self.lower)/max(1., abs(self.upper))
                          if np.isfinite(self.upper) else None),
            converged=closed(self.lower, self.upper, self.o.tolerance),
            optimality_cuts=sum(c.optimality for c in self.cuts),
            feasibility_cuts=sum(not c.optimality for c in self.cuts))

    @property
    def seeds(self):
        pool = sorted((r for r in self.cache.values() if r.feasible), key=lambda r: r.cost)[:16]
        return [r.point[self.b.binary].astype(np.uint8) for r in pool]

    def evaluate(self, x, source):
        b, whole, result = self.b, self.whole, self
        cache, cuts = self.cache, self.cuts
        remaining = whole.remaining
        key = np.asarray(x, np.uint8).tobytes()
        if remaining() <= 0:
            return
        if key in cache:
            return cache[key]
        phase = perf_counter()
        r = dispatch(b, x, remaining(), certified=True) if self.o.paper_sa else dispatch(b, x, remaining())
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
        if r.cut is not None and (not self.o.paper_sa or record_cut(self, r.cut, x, source)):
            cuts.append(r.cut)
        if r.feasible:
            p = r.point
            result.planes.append((float(b.original_obj @ p + b.q @ (p ** 2)), b.crossA @ p - b.crossrhs))
            if r.cost < result.upper:
                result.best, result.upper = r, r.cost
        whole.diagnostics["local_evaluations"].append(dict(outer=self.outer, block=self.index, source=source,
            feasible=r.feasible, cost=r.cost if r.feasible else None, context=self.context))
        return r

    def step(self, seeds, inner):
        b, o, whole, result = self.b, self.o, self.whole, self
        outer, index, cuts = self.outer, self.index, self.cuts
        remaining, evaluate = whole.remaining, self.evaluate
        if remaining() <= 0:
            return
        self.diagnostic["visits"] += 1
        if self.diagnostic["first_visit_seconds"] is None:
            self.diagnostic["first_visit_seconds"] = perf_counter() - whole.started
        for seed in seeds:
            evaluate(seed, "initial")
        if remaining() <= 0:
            return
        phase = perf_counter()
        m, bound = master(b, cuts, remaining())
        if o.paper_sa:
            record_bound(self, m)
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            return
        result.lower = max(result.lower, bound)
        sampling = {"bits": 0, "candidates": 0, "best_energy": None, "seconds": 0.}
        if not closed(result.lower, result.upper, o.tolerance):
            if result.best is None:
                raise RuntimeError("No feasible block seed")
            baseline_u = whole.result.u.copy()
            phase = perf_counter()
            m.lb[-1] = max(m.lb[-1], result.lower - b.constant)
            q = qubo(m, o, result.upper - b.constant)
            whole.timings["qubo"] += perf_counter() - phase
            if whole.progress is not None:
                whole.diagnostics["pending_qubo"] = dict(bits=q.Q.shape[0], nonzeros=q.Q.nnz,
                    max_coefficient=float(abs(q.Q).max()), dense_working_bytes=32*(q.Q.shape[0]+1)**2)
                whole.emit("before_sampling")
            if remaining() <= 0:
                return
            phase = perf_counter()
            whole.diagnostics["sa_calls"] += 1
            self.diagnostic["sa_calls"] += 1
            sample_seed = (int(np.random.SeedSequence([o.seed, outer, inner, index]).generate_state(1)[0] % (2**31-1))
                           if o.rts24_improvements or o.paper_sa else o.seed)
            proposals, sampling = anneal(q, o, remaining(), sample_seed)
            whole.timings["annealing"] += perf_counter() - phase
            sampling.update(outer=outer, block=index, inner=inner, round=inner, seed=sample_seed,
                            raw_candidates=len(proposals), raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals),
                            unique_uvw=len({x.astype(np.uint8).tobytes() for x in proposals}))
            if o.paper_sa:
                sampling_evidence(self, m, q, sampling)
            whole.diagnostics["sampling"].append(sampling)
            whole.emit("sampling")
            del q
            if remaining() <= 0:
                return
            if o.paper_sa:
                process_raw_batch(self, proposals, baseline_u, inner)
            # All samples use the same incumbent snapshot, before this batch's improvements.
            for k, x in enumerate([] if o.paper_sa else proposals):
                if remaining() <= 0:
                    return
                local = np.zeros(len(b.obj)); local[b.binary] = x
                raw = baseline_u.copy(); raw[:, b.times] = local[b.u]
                record = {"proposal_id": f"sa:{whole.diagnostics['sa_calls']}:{k}", "outer": outer, "block": index, "round": inner}
                repaired = whole.prepare(raw, "sa", record)
                if repaired is not None and remaining() > 0:
                    local = evaluate(repaired[b.cols[b.binary]], "sa")
                    record.update(local_dispatch_completed=local is not None,
                                  local_feasible=local.feasible if local is not None else None)
                    whole.accept(repaired, "sa", record)
        if remaining() <= 0:
            return
        phase = perf_counter()
        m, bound = master(b, cuts, remaining())
        if o.paper_sa:
            record_bound(self, m)
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            return
        result.lower = max(result.lower, bound)
        optimality = sum(c.optimality for c in cuts)
        result.history.append(dict(outer=outer, block=index, inner=inner, lower=result.lower,
            upper=result.upper, optimality_cuts=optimality, feasibility_cuts=len(cuts)-optimality, sampling=sampling))
        whole.result.history.append(result.history[-1])
        self.summarize()


def _solve(d, scenarios, o, *, resume_state=None, initial_state=None, progress=None, started_at=None):
    if o.rts24_improvements and resume_state is not None:
        raise ValueError('Improved RTS24 experiments require fresh initialization')
    if resume_state is not None and initial_state is not None:
        raise ValueError("Common initialization and resume are mutually exclusive")
    prior_state = resume_state if resume_state is not None else initial_state
    prior = prior_state["charged_elapsed_seconds"] if prior_state is not None else 0.
    if not np.isfinite(prior) or prior < 0 or (resume_state is not None and prior >= o.time_limit):
        raise ValueError("Invalid or exhausted cumulative resume budget")
    if prior >= o.time_limit:
        raise ValueError("Common initialization exhausts the run budget")
    started = (perf_counter() if started_at is None else started_at)-prior; deadline = started + o.time_limit
    if o.rts24_improvements or o.paper_sa:
        deadline -= 5.
    result, whole = Result(), None
    searches = []
    stage = "INITIALIZATION"

    def remaining():
        return deadline - perf_counter()

    try:
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        full.audit_cuts = o.paper_sa or bool(getattr(progress, 'audit_cuts', False))
        whole = WholeDay(m, full, o, deadline, result, started, progress)
        phase = perf_counter()
        if initial_state is None:
            relaxation = optimize(m, remaining())
        else:
            if initial_state["model_fingerprint"] != whole.diagnostics["model_fingerprint"]:
                raise ValueError("Common initialization model fingerprint mismatch")
            prices = np.asarray(initial_state["relaxation_prices"], float)
            if prices.shape != m.rhs.shape or not np.isfinite(prices).all():
                raise ValueError("Invalid common relaxation prices")
            relaxation = NS(status="OPTIMAL", lower=initial_state["lower_bound"],
                            objval=initial_state["lower_bound"], prices=prices)
            whole.diagnostics["common_initialization"] = dict(sha256=initial_state["sha256"],
                charged_elapsed_seconds=prior)
        whole.timings["initialization"] += perf_counter() - phase
        if remaining() <= 0:
            raise TimeoutError
        if relaxation.status != "OPTIMAL":
            raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
        if hasattr(progress, 'relaxation'):
            progress.relaxation(m, relaxation, perf_counter()-started)
        result.lower_bound = relaxation.lower if whole.certifier is not None or o.paper_sa else relaxation.objval - 1e-5
        if not np.isfinite(result.lower_bound):
            raise RuntimeError("Continuous relaxation has no finite certified lower bound")
        prices = relaxation.prices
        if resume_state is not None:
            stage = "RESUME"
            whole.restore(resume_state)
        elif initial_state is not None:
            x = pack(d, np.asarray(initial_state["u"]))
            whole.accept(x, "initial", {"proposal_id": "initial:common"})
            if whole.best is None or abs(result.objective-initial_state["objective"]) > 1e-4:
                raise ValueError("Common initial schedule failed independent reevaluation")
        else:
            for u in (np.floor(relaxation.x[m.ix.u] + .5), np.ones((d.n_units, d.horizon))):
                record = {"proposal_id": f"initial:{len(whole.diagnostics['proposals'])}"}
                x = whole.prepare(u, "initial", record)
                whole.accept(x, "initial", record)
        if whole.best is None:
            raise RuntimeError("No validated initial schedule")
        whole.emit("initialization")
        if (whole.certifier is not None and (initial_state is not None or not o.sa_enabled)
                and whole.certifier.converged()):
            result.status = "CONVERGED"
        blocks, cross = partition(m, o.block_hours, prices) if o.sa_enabled else ([], np.array([], int))
        planes, previous = [[] for _ in blocks], [[] for _ in blocks]
        first_outer, first_round = whole.resume_cursor["outer"], whole.resume_cursor["round"]
        for outer in (range(first_outer, o.max_outer + 1) if o.sa_enabled and result.status != "CONVERGED" else []):
            if remaining() <= 0:
                raise TimeoutError
            blocks, _ = partition(m, o.block_hours, prices)
            if o.paper_sa:
                record_state(whole, prices, outer)
            searches = [BlockSearch(b, o, outer, j + 1, whole, planes[j]) for j, b in enumerate(blocks)]
            if whole.advice is not None and whole.advice.state["phase"] == "llm" and not o.rts24_improvements:
                stage = "ADVISOR_RESUME"
                whole.advice.run_batch(searches)
                if whole.certifier.converged():
                    result.status = "CONVERGED"
                    break
            for inner in range(first_round if outer == first_outer else 1, o.max_inner + 1):
                before = result.objective
                lower_before, sa_before = result.lower_bound, whole.diagnostics["sa_calls"]
                before_count, before_calls = len(whole.diagnostics["evaluations"]), result.llm_calls
                before_opt = sum(c.optimality for search in searches for c in search.cuts)
                before_feas = sum(not c.optimality for search in searches for c in search.cuts)
                round_record = dict(outer=outer, round=inner, complete=False, visited_blocks=[],
                                    skipped_closed_blocks=[], started_seconds=perf_counter()-started)
                whole.diagnostics["rounds"].append(round_record)
                for j, search in enumerate(searches):
                    if o.rts24_improvements:
                        whole.advice.service(searches)
                    if remaining() <= 0:
                        raise TimeoutError
                    if closed(search.lower, search.upper, o.tolerance):
                        round_record["skipped_closed_blocks"].append(j + 1)
                        continue
                    b = search.b
                    stage = f"BLOCK_{outer}_{j + 1}_ROUND_{inner}"
                    seeds = [whole.best.point[b.cols[b.binary]], *(previous[j] if inner == 1 else []),
                             *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
                    search.step(seeds, inner)
                    # Every term belongs to the current multiplier vector, including unvisited blocks.
                    if whole.certifier is None:
                        result.lower_bound = max(result.lower_bound, float(sum(x.lower for x in searches)))
                    if remaining() <= 0:
                        raise TimeoutError
                    round_record["visited_blocks"].append(j + 1)
                    if search.best is not None:
                        raw = result.u.copy(); raw[:, b.times] = search.best.point[b.u]
                        record = {"proposal_id": f"pool:{outer}:{inner}:{j+1}",
                                  "outer": outer, "round": inner, "block": j + 1}
                        x = whole.prepare(raw, "pool", record)
                        whole.accept(x, "sa_pool_repaired" if o.paper_sa else "pool", record)
                    if o.verbose:
                        print(f"outer={outer} round={inner} block={j + 1}/{len(blocks)} "
                              f"local=[{search.lower:.6f}, {search.upper:.6f}] "
                              f"full={result.objective:.6f} LB={result.lower_bound:.6f}", flush=True)
                    if remaining() <= 0:
                        raise TimeoutError
                round_record.update(complete=True, sa_completed_seconds=perf_counter()-started,
                                    incumbent_before=before, incumbent_after_sa=result.objective)
                whole.diagnostics["complete_rounds"] += 1
                iteration = dict(iteration=len(whole.diagnostics["iterations"])+1, outer=outer, round=inner,
                    incumbent=result.objective,
                    optimality_cuts_added=sum(c.optimality for x in searches for c in x.cuts)-before_opt,
                    feasibility_cuts_added=sum(not c.optimality for x in searches for c in x.cuts)-before_feas,
                    new_candidates=len(whole.diagnostics["evaluations"])-before_count, llm_failed=False)
                whole.diagnostics["iterations"].append(iteration)
                all_closed = all(closed(x.lower, x.upper, o.tolerance) for x in searches)
                whole.resume_cursor = (dict(outer=outer+1, round=1) if all_closed or inner == o.max_inner
                                       else dict(outer=outer, round=inner+1))
                if whole.certifier is not None:
                    stage = "CERTIFICATION"
                    whole.certifier.step()
                    stage = "ADVISOR"
                    whole.advice.after_round(before, lower_before, sa_before, searches, outer, inner,
                                             not (outer == o.max_outer and inner == o.max_inner))
                else:
                    stage = f"ADVISOR_{outer}_ROUND_{inner}"
                    whole.after_round(before, searches, outer, inner, not all_closed)
                iteration.update(incumbent=result.objective,
                    optimality_cuts_added=sum(c.optimality for x in searches for c in x.cuts)-before_opt,
                    feasibility_cuts_added=sum(not c.optimality for x in searches for c in x.cuts)-before_feas,
                    new_candidates=len(whole.diagnostics["evaluations"])-before_count,
                    llm_failed=any(row["status"] != "OK" for row in result.llm_history[before_calls:]))
                round_record.update(ended_seconds=perf_counter()-started, incumbent=result.objective,
                                    llm_calls=result.llm_calls-before_calls)
                whole.emit("round")
                for search in searches:
                    search.summarize()
                if remaining() <= 0:
                    raise TimeoutError
                if ((whole.certifier is not None and whole.certifier.converged())
                        or (o.paper_sa and closed(result.lower_bound, result.objective, o.tolerance))):
                    result.status = "CONVERGED"
                    break
                if all_closed:
                    break
            if result.status == "CONVERGED":
                break
            previous = [search.seeds for search in searches]
            stage = "COORDINATION"
            phase = perf_counter()
            values, upper = coordinate(planes, m.eq[cross], result.objective, remaining())
            whole.timings["coordination"] += perf_counter() - phase
            if remaining() <= 0:
                raise TimeoutError
            result.coordination_upper = upper
            result.outer_history.append({"outer": outer, "lower": result.lower_bound, "coordination_upper": upper,
                                         "complete_rounds": inner,
                                         "unconverged_blocks": [x.index for x in searches
                                             if not closed(x.lower, x.upper, o.tolerance)]})
            whole.diagnostics["coordination_count"] += 1
            if o.paper_sa:
                whole.diagnostics.setdefault('paper_coordination', []).append(dict(
                    **result.outer_history[-1], approximate=not all_closed,
                    elapsed_seconds=perf_counter()-started))
            whole.emit("coordination")
            if whole.certifier is not None:
                dual_lower = max(relaxation.lower, float(sum(search.lower for search in searches)))
                result.outer_history[-1]["dual_lower"] = dual_lower
                result.outer_history[-1]["dual_converged"] = closed(dual_lower, upper, o.tolerance, dual_lower)
            elif not o.paper_sa and closed(result.lower_bound, upper, o.tolerance, result.lower_bound):
                result.status = "DUAL_CONVERGED"
                break
            prices = np.zeros(len(m.rhs)); prices[cross] = values
            if o.rts24_improvements:
                whole.check_repricing(prices, outer)
        if whole.certifier is not None and result.status != "CONVERGED":
            stage = "CERTIFICATION"
            while remaining() > 0 and not whole.certifier.converged():
                if o.rts24_improvements:
                    whole.advice.service(searches)
                whole.certifier.step()
            if remaining() <= 0:
                raise TimeoutError
            result.status = "CONVERGED"
    except Exception as error:
        result.status = "TIME_LIMIT" if remaining() <= 0 or isinstance(error, TimeoutError) else f"{stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    finally:
        if whole is not None and whole.certifier is not None:
            whole.advice.close()
            whole.certifier.close()
    for search in searches:
        search.summarize()
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if o.bound_mode == "integer_benders" and result.lower_bound > result.objective + 1e-5:
            result.status, result.message = "CERTIFICATION_FAILED", "Lower bound exceeds validated cost"
            result.relative_gap = (result.objective - result.lower_bound) / max(1., abs(result.objective))
        else:
            if result.lower_bound > result.objective + 1e-4:
                raise RuntimeError("Lower bound exceeds validated cost")
            result.relative_gap = max(0., result.objective - result.lower_bound) / max(1., abs(result.objective))
    result.elapsed_seconds = perf_counter() - started
    if whole is not None:
        whole.timings["other"] = max(0., result.elapsed_seconds - sum(whole.timings.values()))
        result.diagnostics.update(evaluation_count=len(whole.cache), llm_call_count=result.llm_calls,
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit),
                                  resume_cursor=whole.resume_cursor)
        whole.emit("final")
    if o.verbose:
        print(f"{result.status}: cost={result.objective:.6f}, lower={result.lower_bound:.6f}, "
              f"gap={100 * result.relative_gap:.4f}%, elapsed={result.elapsed_seconds:.2f}s")
    return result


def solve(d, scenarios, o, *, resume_state=None, initial_state=None, progress=None, started_at=None):
    args = dict(resume_state=resume_state, initial_state=initial_state, progress=progress, started_at=started_at)
    if not o.paper_sa:
        return _solve(d, scenarios, o, **args)
    if resume_state is not None or initial_state is not None:
        raise ValueError('Paper SA requires fresh initialization')
    calls = []
    def record(event):
        calls.append(event)
        if hasattr(progress, 'optimizer_event'):
            progress.optimizer_event(event)
    with continuous_only(record):
        result = _solve(d, scenarios, o, **args)
    result.diagnostics['optimizer_calls'] = calls
    return result
