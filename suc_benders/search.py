"""Time-block SA and shared whole-day candidate evaluation."""
# 分块搜索只产生候选；全天修复、调度和独立校验通过后才能更新 UB。
import hashlib
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .anneal import anneal, SA_BACKEND, SA_INITIALIZATION
from .benders import discrete_feasible, dispatch, hard_rows
from .model import fingerprint, relaxed_rows
from .optimization import optimize
from .qubo import search_qubo, fixed_binary
from .strategies import strategies_for_options
from .validation import fixed_cost, capacity, commitment, commitment_hash, pack, repair, unpack, validate, transition_allowed, reachable_shortfall


def _default_sampler(q, o, seconds, seeds):
    """Resolve the module-level sampler at call time so existing patches still work."""
    return anneal(q, o, seconds, seeds)


# 全天割池先进先出；每个块只编码小工作集，控制辅助位和稠密耦合规模。
CUT_LIMIT = 4096
QUBO_FEASIBILITY_CUTS = 64
# shortcut: 仅搜索 8 条割的子集包络；若旧候选反复被低估，再评估更大的工作集。
QUBO_OPTIMALITY_CUTS = 8


# 全天 u/v/w 二进制向量的缓存键。
def _cache_key(x):
    return x.astype(np.uint8).tobytes()


# 辅助 SA 工作配额耗尽的内部信号；与整个求解时间耗尽区分，配额耗尽记为 SEARCH_EXHAUSTED。
class SABudgetExhausted(Exception):
    pass


# 各块共用的全天方案、调度缓存、数值反馈和预算。
class WholeDay:
    """Shared incumbent, measured feedback and deadline across all time blocks."""

    # 绑定输入模型与搜索截止时间，建立按来源统计的诊断和计时器。
    def __init__(self, m, full, o, deadline, result, started, progress=None, strategies=None):
        self.m, self.full, self.o, self.deadline = m, full, o, deadline
        self.d, self.scenarios, self.result, self.started = m.d, m.scenarios, result, started
        self.strategies = strategies or strategies_for_options(o, sampler=_default_sampler)
        self.best = None
        self.cache = {}
        self.cuts = []
        self.cut_cache_keys = {}
        self.evaluated_commitments = set()
        self.cut_totals = dict(optimality=0, feasibility=0)
        self.diagnostics = result.diagnostics
        self.diagnostics.update(sa_calls=0, evaluations=[], proposals=[], sampling=[], iterations=[],
                                blocks=[], rounds=[], cuts_dropped=0, stored_cuts=0,
                                cache_invalidations=0, re_evaluations=0,
                                dynamic_constraints_added=0,
                                cut_totals=self.cut_totals,
                                complete_rounds=0, timings={name: 0. for name in (
                                    "initialization", "qubo", "annealing", "repair",
                                    "subproblems", "validation")})
        self.timings = self.diagnostics["timings"]
        self.progress = progress
        self.diagnostics["model_fingerprint"] = fingerprint(m)
        self.sa_started = None
        self.diagnostics['sa_budget_seconds_used'] = 0.
        self.diagnostics.update(search_mode='time_blocks', sa_backend=SA_BACKEND, sa_initialization=SA_INITIALIZATION,
                                constraint_policy=self.strategies.constraint_policy.name,
                                cut_selector=getattr(self.strategies.cut_selector, '__name__',
                                                     type(self.strategies.cut_selector).__name__),
                                sampler=getattr(self.strategies.sampler, '__name__',
                                                type(self.strategies.sampler).__name__))
        phase = perf_counter()
        self.fixed = fixed_binary(hard_rows(full, reachability=True))
        start, stop = transition_allowed(self.d)
        fixed_u = self.fixed_u = self.fixed[full.u]
        self.diagnostics['reachability'] = dict(index_base=1,
            movable_units=(np.flatnonzero(np.isnan(fixed_u).any(axis=1))+1).tolist(),
            frozen_units=(np.flatnonzero(np.isfinite(fixed_u).all(axis=1))+1).tolist(),
            fixed_u=np.where(np.isfinite(fixed_u), fixed_u, -1).astype(int).tolist(),
            fixed_u_unknown=-1, startup_allowed=start.tolist(), shutdown_allowed=stop.tolist())
        self.timings['qubo'] += perf_counter()-phase

    def add_cut(self, cut, cache_key=None, *, protected=None):
        if not cut.certified or cut.beta.shape != (len(self.full.binary),) or not (
                np.isfinite(cut.alpha) and np.isfinite(cut.beta).all()):
            raise ValueError("Whole-day search requires a finite certified full-day cut")
        self.cuts.append(cut)
        self.cut_cache_keys[id(cut)] = cache_key
        self.cut_totals["optimality" if cut.optimality else "feasibility"] += 1
        if protected is None and self.best is not None:
            protected = self.best.cut
        if len(self.cuts) > CUT_LIMIT:
            index = next(k for k, old in enumerate(self.cuts) if old is not protected)
            dropped = self.cuts.pop(index)
            owner = self.cut_cache_keys.pop(id(dropped))
            if owner in self.cache:
                del self.cache[owner]
                self.diagnostics['cache_invalidations'] += 1
            self.diagnostics["cuts_dropped"] += 1
        self.diagnostics["stored_cuts"] = len(self.cuts)
        return id(cut) in self.cut_cache_keys

    # 返回已记账的 SA 辅助工作秒数，加上当前尚未退出计时作用域的耗时。
    def sa_used(self):
        return self.diagnostics['sa_budget_seconds_used'] + (
            0. if self.sa_started is None else perf_counter()-self.sa_started)

    # 将建模、采样、修复、调度或协调整段工作记入 SA 配额，并临时收紧数值调用截止时间。
    # 退出时恢复搜索 deadline；配额耗尽转为 SABudgetExhausted，以便收尾在途请求。
    @contextmanager
    def sa_work(self):
        """Charge the complete auxiliary search operation and cap its numeric calls."""
        quota = self.o.sa_budget_seconds
        if quota is None:
            yield
            return
        used = self.sa_used()
        if used >= quota:
            raise SABudgetExhausted
        original = self.deadline
        self.sa_started = perf_counter()
        limit = self.sa_started + quota-used
        self.deadline = min(original, limit)
        try:
            yield
            if perf_counter() >= limit and limit < original:
                raise SABudgetExhausted
        except TimeoutError:
            if perf_counter() >= limit and limit < original:
                raise SABudgetExhausted from None
            raise
        finally:
            self.diagnostics['sa_budget_seconds_used'] += perf_counter()-self.sa_started
            self.sa_started = None
            self.deadline = original

    # 返回当前截止时间前的剩余秒数；非正值表示后续结果不得再入账。
    def remaining(self):
        return self.deadline - perf_counter()

    # 若提供 progress 回调则发送事件和当前 WholeDay；核心求解器自身不在此处落盘。
    def emit(self, event):
        if self.progress is not None:
            self.progress(self, event)

    def initialize(self, relaxation):
        target = np.floor(relaxation.x[self.m.ix.u] + .5)
        for u in (target, np.ones((self.d.n_units, self.d.horizon))):
            record = {"proposal_id": f"initial:{len(self.diagnostics['proposals'])}"}
            self.accept(self.prepare(u, "initial", record), "initial", record)
        if self.remaining() <= 0:
            raise TimeoutError
        if self.best is not None:
            return

        phase = perf_counter()
        accounted = sum(self.timings[name] for name in ("repair", "subproblems", "validation"))
        info = dict(source="initialization_mip", status="RUNNING", outcome="no_candidate")
        self.diagnostics["initialization_repair"] = info
        try:
            deadline = min(phase+30., self.deadline-5.)
            if deadline <= phase:
                info["status"] = "INSUFFICIENT_BUDGET"
                raise RuntimeError("Initialization repair: INSUFFICIENT_BUDGET for projection and validation")
            projection = NS(**vars(self.m))
            projection.obj = np.zeros_like(self.m.obj)
            projection.obj[self.m.ix.u] = 1-2*target
            projection.q = np.zeros_like(self.m.q)
            projection.constant = 0.
            if self.o.verbose:
                print("[初始化修复] 常规候选不可行；启动最多 30 秒整数可行性投影，"
                      "首个可行解返回后独立调度和校验。", flush=True)
            seconds = deadline-perf_counter()
            if seconds <= 0:
                info.update(status="TIME_LIMIT", outcome="projection_deadline")
                raise RuntimeError("Initialization repair: TIME_LIMIT while constructing the projection")
            raw = optimize(projection, seconds, integral=True,
                           mip_focus=1, solution_limit=1, int_feas_tol=1e-9)
            info["status"] = raw.status
            if self.remaining() <= 0:
                info["outcome"] = "deadline"
                raise TimeoutError("Initialization repair exceeded the total deadline")
            if perf_counter() > deadline:
                info["outcome"] = "projection_deadline"
                raise RuntimeError("Initialization repair: TIME_LIMIT; late projection candidate discarded")
            if raw.status not in ("OPTIMAL", "SOLUTION_LIMIT", "TIME_LIMIT") or raw.x is None:
                raise RuntimeError(f"Initialization repair: {raw.status}; no usable feasible candidate")
            u = np.rint(raw.x[self.m.ix.u])
            record = {"proposal_id": "initialization_mip:1"}
            accepted = self.accept(self.prepare(u, "initialization_mip", record), "initialization_mip", record)
            info["outcome"] = record["outcome"]
            if self.remaining() <= 0:
                raise TimeoutError("Initialization repair validation exceeded the total deadline")
            if not accepted:
                raise RuntimeError(f"Initialization repair candidate rejected: {record['outcome']}")
            info.update(hamming_u=int(np.count_nonzero(self.result.u != target)),
                        objective=self.result.objective)
        except Exception as error:
            if info["status"] == "RUNNING":
                info["status"] = "ERROR"
            info["message"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            info["elapsed_seconds"] = perf_counter()-phase
            # 修复、调度和校验已有独立计时，初始化只补记剩余部分。
            charged = sum(self.timings[name] for name in ("repair", "subproblems", "validation"))-accounted
            self.timings["initialization"] += max(0., info["elapsed_seconds"]-charged)
            self.emit("initialization_repair")

    # 对全天原始 u 做形状/有限值检查及启停/容量修复，记录变化、来源和重复状态。
    # 返回重建的 u/v/w 向量或 None；修复可能涉及块外小时，之后仍须完成连续调度和验证。
    # 每条记录只存统计所需的原始方案哈希与固定位违例数，不保存方案矩阵。
    def prepare(self, raw, source, record):
        """Repair a whole-day u proposal before full-day dispatch and validation."""
        values = np.asarray(raw, dtype='<f8')
        known = np.isfinite(self.fixed_u)
        record.update(origin=source, raw_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                      raw_fixed_bit_violations=int(np.count_nonzero(values[known] != self.fixed_u[known]))
                      if values.shape == self.fixed_u.shape else 0)
        self.diagnostics["proposals"].append(record)
        if self.remaining() <= 0:
            record["outcome"] = "deadline"
            return None
        raw = np.asarray(raw, float)
        if raw.shape != (self.d.n_units, self.d.horizon) or not np.isfinite(raw).all():
            record["outcome"] = "invalid_matrix"
            return None
        raw_x = pack(self.d, raw)
        z = commitment(self.d, raw_x)
        required = (1 + self.d.reserve_fraction) * np.max([s.load for s in self.scenarios], axis=0)
        record["raw_discrete_feasible"] = bool(discrete_feasible(self.full, raw_x)
            and np.array_equal(raw_x[np.isfinite(self.fixed)], self.fixed[np.isfinite(self.fixed)])
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
        feasible = bool(discrete_feasible(self.full, x)
            and (capacity(self.d, z.u, z.v, z.w).sum(axis=0) >= required - 1e-6).all()
            and (np.sum(self.d.pmin[:, None]*z.u, axis=0)
                 <= np.min([s.load for s in self.scenarios], axis=0) + 1e-6).all())
        record.update(commitment_id=commitment_hash(z),
                      hamming_u=int(np.count_nonzero(z.u != raw)), repaired_discrete_feasible=feasible,
                      duplicate=_cache_key(x) in self.cache, outcome="repaired")
        if not feasible:
            record["outcome"] = "repair_discrete_infeasible"
            return None
        # 计入启停和爬坡链后可达出力仍低于负荷的方案必然调度不可行，跳过调度。
        if reachable_shortfall(self.d, self.scenarios, z.u):
            record["outcome"] = "reachable_infeasible"
            return None
        return x

    def search_statistics(self):
        """Count whole-day proposals and first validations, never block feasibility."""
        statistics = {}
        for source in ('sa_raw', 'sa'):
            proposals = [r for r in self.diagnostics['proposals'] if r['origin'] == source]
            evaluations = [r for r in self.diagnostics['evaluations'] if r['source'] == source]
            raw = {r['raw_sha256'] for r in proposals}
            repaired = {r['commitment_id'] for r in proposals if r.get('repaired_discrete_feasible')}
            outcomes = {}
            for r in proposals:
                outcome = r['outcome']; outcomes[outcome] = outcomes.get(outcome, 0)+1
            statistics[source] = dict(proposals=len(proposals), raw_unique=len(raw), repaired_unique=len(repaired),
                raw_duplicate_rate=1-len(raw)/len(proposals) if proposals else 0.,
                repaired_duplicate_rate=1-len(repaired)/sum(bool(r.get('repaired_discrete_feasible')) for r in proposals)
                    if repaired else 0.,
                raw_discrete_violations=sum(r.get('raw_discrete_feasible') is False for r in proposals),
                raw_fixed_bit_violations=sum(r['raw_fixed_bit_violations'] for r in proposals),
                repair_bits=sum(r.get('hamming_u', 0) for r in proposals),
                duplicate_rate=outcomes.get('duplicate', 0)/len(proposals) if proposals else 0.,
                outcomes=outcomes,
                new_feasible=sum(r['feasible'] and not r['reevaluated'] for r in evaluations),
                incumbent_decrease=sum(r['incumbent_before']-r['incumbent'] for r in evaluations
                                       if r['improved'] and np.isfinite(r['incumbent_before'])))
        statistics['new_feasible'] = sum(row['new_feasible'] for row in statistics.values())
        return statistics

    # 对全天二进制方案去重、做连续调度和独立物理/费用校验，记录实测结果及可用认证割。
    # 只有真实费用更低的可行方案更新 incumbent；命中缓存的重复候选不重复贡献降本。
    def accept(self, x, source, record=None):
        if x is None:
            if record is not None:
                record.setdefault("outcome", "no_candidate")
            return False
        if self.remaining() <= 0:
            if record is not None:
                record["outcome"] = "deadline"
            return False
        key = _cache_key(x)
        if key in self.cache:
            if record is not None:
                record.update(outcome="duplicate", duplicate=True, cache_hit=True, incumbent_decrease=0.)
            return self.cache[key]["feasible"]
        phase = perf_counter()
        r = dispatch(self.full, x, self.remaining(), certified=True)
        self.timings["subproblems"] += perf_counter() - phase
        if self.remaining() <= 0 or r.status == "TIME_LIMIT":
            if record is not None:
                record["outcome"] = "dispatch_timeout"
            return False
        if r.status not in ("OPTIMAL", "INFEASIBLE", "DISCRETE_INFEASIBLE"):
            # SUBOPTIMAL 等数值状态只拒绝该候选且不缓存，求解继续。
            self.diagnostics["dispatch_failures"] = self.diagnostics.get("dispatch_failures", 0) + 1
            if record is not None:
                record["outcome"] = "dispatch_failed"
            return False
        before = self.result.objective
        # plan 先是 u/v/w；可行时换成含各场景调度的完整解，哈希与费用都取自当前的 plan。
        plan = commitment(self.d, x)
        binary_cost = fixed_cost(self.d, plan.u, plan.w)
        report, scenario_costs = None, {}
        if r.feasible:
            plan = unpack(self.m, r.point)
            phase = perf_counter()
            report = validate(self.d, self.scenarios, plan)
            self.timings["validation"] += perf_counter() - phase
            scenario_costs = {s.name: s.objective for s in plan.scenarios}
        if self.remaining() <= 0:
            if record is not None:
                record["outcome"] = "validation_deadline"
            return False
        improved = r.feasible and plan.objective < before
        # 先保护即将采用的新割；上限为 1 时也不能淘汰新 incumbent 的反馈。
        retained = r.cut is None or self.add_cut(r.cut, key, protected=r.cut if improved else None)
        identity = commitment_hash(plan)
        reevaluated = identity in self.evaluated_commitments
        entry = dict(source=source, feasible=r.feasible, dispatch_status=r.status,
                     objective=plan.objective if r.feasible else None,
                     fixed_cost=binary_cost, expected_fuel_cost=plan.objective-binary_cost if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=identity, reevaluated=reevaluated,
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        if record is not None:
            entry["proposal_id"] = record.get("proposal_id")
            record["incumbent_decrease"] = float(before-plan.objective) if improved and np.isfinite(before) else 0.
        if retained:
            self.cache[key] = entry
        self.evaluated_commitments.add(identity)
        self.diagnostics['re_evaluations'] += int(reevaluated)
        self.diagnostics["evaluations"].append(entry)
        if record is not None:
            record.update(outcome="accepted" if r.feasible else "infeasible", feasible=r.feasible,
                          continuous_dispatch_completed=True, validation_completed=r.feasible)
        if improved:
            self._adopt_incumbent(r, plan, report)
        self.emit("evaluation")
        return r.feasible

    # 以已校验的可行解替换 incumbent（含各场景调度）。
    def _adopt_incumbent(self, r, plan, report):
        self.best = r
        for name in ("objective", "u", "v", "w", "scenarios"):
            setattr(self.result, name, getattr(plan, name))
        self.result.validation = report


# 时间块仅产生候选；费用与可行性反馈来自共享全天割池。
class BlockSearch:
    """Block proposals using full-day cuts projected at a frozen incumbent."""

    def __init__(self, b, o, index, whole):
        self.b, self.o, self.index, self.whole = b, o, index, whole
        self.columns = b.cols[b.binary]
        self.fixed = whole.fixed[self.columns].copy()
        self.constraint_rows = []
        self.constraint_signatures = set()
        self.hard = hard_rows(b, fixed=self.fixed)
        m = whole.m
        rows = m.A[b.cross].tocsr()
        inside = np.zeros(rows.shape[1], bool); inside[self.columns] = True
        keep = ~relaxed_rows(m, b.cross) & (rows[:, inside].getnnz(axis=1) > 0)
        rows = rows[keep]
        self.boundary = NS(inner=rows[:, self.columns].tocsr(),
                           outer=(rows @ sparse.diags((~inside).astype(float))).tocsr(),
                           rhs=m.rhs[b.cross][keep], eq=m.eq[b.cross][keep])
        self.diagnostic = dict(block=index, hours=(b.times + 1).tolist(), visits=0,
                               sa_calls=0, first_visit_seconds=None)
        whole.diagnostics["blocks"].append(self.diagnostic)

    def _observe_constraint(self, candidate, reason):
        policy = self.whole.strategies.constraint_policy
        for row in policy.observe(self.b, candidate, reason):
            signature = row.signature()
            if signature in self.constraint_signatures:
                continue
            self.constraint_signatures.add(signature)
            self.constraint_rows.append(row)
            self.diagnostic['dynamic_constraint_rows'] = len(self.constraint_rows)
            self.whole.diagnostics['dynamic_constraints_added'] = (
                self.whole.diagnostics.get('dynamic_constraints_added', 0) + 1)

    def boundary_rows(self, point):
        ring = self.boundary
        rhs = ring.rhs - ring.outer @ point
        return [(ring.inner.indices[ring.inner.indptr[k]:ring.inner.indptr[k+1]],
                 ring.inner.data[ring.inner.indptr[k]:ring.inner.indptr[k+1]], rhs[k], ring.eq[k])
                for k in range(len(rhs))]

    def qubo_cuts(self, point):
        cuts = self.whole.cuts
        anchor = point[self.whole.full.binary]
        values = [c.alpha + c.beta @ anchor for c in cuts]
        optimality = [k for k, cut in enumerate(cuts) if cut.optimality]
        feasibility = [k for k, cut in enumerate(cuts) if not cut.optimality]
        chosen, selected_feasibility = self.whole.strategies.cut_selector(
            cuts, values, self.columns, QUBO_OPTIMALITY_CUTS, QUBO_FEASIBILITY_CUTS)
        # 原始全天割不变；每批按块外快照重新投影截距。
        outside = anchor.copy(); outside[self.columns] = 0.
        projected = [NS(alpha=cuts[k].alpha + cuts[k].beta @ outside,
                        beta=cuts[k].beta[self.columns], optimality=cuts[k].optimality,
                        certified=cuts[k].certified) for k in chosen + selected_feasibility]
        counts = dict(optimality_cuts_stored=len(optimality), optimality_cuts_selected=len(chosen),
                      feasibility_cuts_stored=len(feasibility), feasibility_cuts_encoded=len(selected_feasibility))
        return projected, counts

    def step(self, k):
        b, o, whole = self.b, self.o, self.whole
        if whole.remaining() <= 0:
            return
        self.diagnostic["visits"] += 1
        if self.diagnostic["first_visit_seconds"] is None:
            self.diagnostic["first_visit_seconds"] = perf_counter() - whole.started
        baseline = whole.best.point.copy()
        anchor_id, anchor_cost = commitment_hash(whole.result), whole.result.objective
        phase = perf_counter()
        boundary = self.boundary_rows(baseline)
        cuts, counts = self.qubo_cuts(baseline)
        q = search_qubo(self.hard, b, cuts, baseline[self.columns], boundary, o,
                        constraint_policy=whole.strategies.constraint_policy,
                        extra_rows=self.constraint_rows)
        counts.update(optimality_cuts_encoded=q.envelope.cuts,
                      optimality_cuts_dominated=counts['optimality_cuts_selected']-q.envelope.cuts,
                      constraint_policy=q.constraint_policy, constraint_rows=q.constraint_rows,
                      local_objective_bound=q.local_objective_bound,
                      dynamic_constraint_rows=len(self.constraint_rows))
        whole.timings["qubo"] += perf_counter() - phase
        self.diagnostic.update(counts)
        if whole.progress is not None:
            whole.diagnostics["pending_qubo"] = dict(bits=q.Q.shape[0], nonzeros=q.Q.nnz,
                max_coefficient=float(np.abs(q.Q.data).max(initial=0.)))
            whole.emit("before_sampling")
        if whole.remaining() <= 0:
            return
        phase = perf_counter()
        whole.diagnostics["sa_calls"] += 1
        self.diagnostic["sa_calls"] += 1
        sample_seeds = [int(np.random.SeedSequence([o.seed, whole.diagnostics["sa_calls"], j]).generate_state(1)[0])
                        for j in range(o.sa_reads)]
        proposals, sampling = whole.strategies.sampler(q, o, whole.remaining(), sample_seeds)
        whole.timings["annealing"] += perf_counter() - phase
        sampling.update(block=self.index, round=k, seed=sample_seeds, boundary_rows=len(boundary),
                        **counts, anchor=anchor_id, anchor_cost=anchor_cost,
                        raw_candidates=len(proposals), raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals),
                        unique_uvw=len({x.astype(np.uint8).tobytes() for x in proposals}))
        whole.diagnostics["sampling"].append(sampling)
        whole.emit("sampling")
        del q
        # 整批都拼接到采样前快照；中途更新 UB 不改变后续候选的块外方案。
        for j, x in enumerate(proposals):
            if whole.remaining() <= 0:
                return
            candidate = baseline[whole.full.binary].copy(); candidate[self.columns] = x
            z = commitment(whole.d, candidate)
            known = np.isfinite(whole.fixed)
            discrete = bool(discrete_feasible(whole.full, candidate)
                            and np.array_equal(candidate[known], whole.fixed[known]))
            proposal_id = f"sa:{whole.diagnostics['sa_calls']}:{j}"
            record = dict(proposal_id=proposal_id, block=self.index, round=k, origin='sa_raw',
                          raw_sha256=hashlib.sha256(np.asarray(z.u, dtype='<f8').tobytes()).hexdigest(),
                          raw_fixed_bit_violations=int(np.count_nonzero(candidate[known] != whole.fixed[known])),
                          raw_discrete_feasible=discrete, outcome='discrete_infeasible')
            whole.diagnostics['proposals'].append(record)
            if discrete:
                accepted = whole.accept(candidate, 'sa_raw', record)
                if record.get('outcome') == 'duplicate':
                    self._observe_constraint(x, 'repeated')
                if accepted:
                    continue
            else:
                self._observe_constraint(x, 'discrete_infeasible')
            repaired_record = dict(proposal_id=proposal_id + ':repair', block=self.index, round=k)
            repaired = whole.prepare(z.u, 'sa', repaired_record)
            whole.accept(repaired, 'sa', repaired_record)
            if repaired_record.get('outcome') in {'duplicate', 'repair_discrete_infeasible', 'reachable_infeasible'}:
                self._observe_constraint(x, 'repair_infeasible')
        if whole.remaining() <= 0:
            return
        whole.result.history.append(dict(block=self.index, round=k, incumbent=whole.result.objective,
                                         optimality_cuts=whole.cut_totals['optimality'],
                                         feasibility_cuts=whole.cut_totals['feasibility'], sampling=sampling))
