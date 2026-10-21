"""Time-block SA and shared whole-day candidate evaluation."""
# 分块搜索只产生候选；全天修复、调度和独立校验通过后才能更新 UB。
import hashlib
from copy import deepcopy
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .anneal import anneal, SA_BACKEND, SA_INITIALIZATION
from .benders import discrete_feasible, dispatch, hard_rows
from .context import CandidateEvaluation, LLMContextInput, build_llm_context
from .llm import MAX_CANDIDATES
from .upper_advisor import UpperAdvisor, summarize
from .model import fingerprint, relaxed_rows
from .optimization import optimize
from .qubo import search_qubo, fixed_binary
from .validation import fixed_cost, capacity, commitment, commitment_hash, pack, repair, unpack, validate, transition_allowed, transition_violations, reachable_shortfall


# LLM 来源的可行方案池保留费用最低的 16 个。
ADVISOR_POOL_SIZE = 16
# 块搜索贯穿整个运行：每块至多保存 CUT_LIMIT 条块内割（先进先出），每个搜索 QUBO 只编码
# anchor 处松弛量最小的 QUBO_FEASIBILITY_CUTS 条可行割，使 QUBO 规模与内存不随轮数增长。
CUT_LIMIT = 4096
QUBO_FEASIBILITY_CUTS = 64


# 全天 u/v/w 二进制向量的缓存键。
def _cache_key(x):
    return x.astype(np.uint8).tobytes()


# 辅助 SA 工作配额耗尽的内部信号；与整个求解时间耗尽区分，以便收尾已有 LLM 工作。
class SABudgetExhausted(Exception):
    pass


# 各块共用的全天方案、调度缓存、数值反馈、预算和 LLM 管理状态。
class WholeDay:
    """Shared incumbent, measured feedback and deadline across all time blocks."""

    # 绑定输入模型与搜索截止时间，建立按来源统计的诊断和计时器。
    def __init__(self, m, full, o, deadline, result, started, progress=None):
        self.m, self.full, self.o, self.deadline = m, full, o, deadline
        self.d, self.scenarios, self.result, self.started = m.d, m.scenarios, result, started
        self.best, self.dispatches = None, []
        self.cache, self.advisor_pool = {}, []
        self.feedback = None
        self.diagnostics = result.diagnostics
        self.diagnostics.update(sa_calls=0, evaluations=[], proposals=[], sampling=[], iterations=[],
                                local_evaluations=[], llm_background_seconds=0., blocks=[], rounds=[],
                                complete_rounds=0, timings={name: 0. for name in (
                                    "initialization", "qubo", "annealing", "repair",
                                    "local_dispatch", "subproblems", "validation", "context", "llm")})
        self.timings = self.diagnostics["timings"]
        self.progress = progress
        self.diagnostics["model_fingerprint"] = fingerprint(m)
        self.advice = UpperAdvisor(self)
        self.sa_started = None
        self.diagnostics['sa_budget_seconds_used'] = 0.
        self.diagnostics.update(search_mode='time_blocks', sa_backend=SA_BACKEND, sa_initialization=SA_INITIALIZATION)
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
    # 每条记录只存统计所需的原始方案哈希与固定位违例数；只有 LLM 提议保留 raw_u/repaired_u 矩阵，
    # 供下一次调用的反馈列出改动区间。
    def prepare(self, raw, source, record):
        """Repair a whole-day u proposal; the caller then evaluates its block projection."""
        values = np.asarray(raw, dtype='<f8')
        known = np.isfinite(self.fixed_u)
        record.update(origin=source, raw_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                      raw_fixed_bit_violations=int(np.count_nonzero(values[known] != self.fixed_u[known]))
                      if values.shape == self.fixed_u.shape else 0)
        if source == 'llm':
            record['raw_u'] = values.tolist()
        self.diagnostics["proposals"].append(record)
        if self.remaining() <= 0:
            record["outcome"] = "deadline"
            return None
        raw = np.asarray(raw, float)
        if raw.shape != (self.d.n_units, self.d.horizon) or not np.isfinite(raw).all():
            record["outcome"] = "invalid_matrix"
            return None
        if source == 'llm' and np.isin(raw, [0, 1]).all():
            record['raw_constraint_violations'] = transition_violations(self.d, raw)
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
        if source == 'llm':
            record['repaired_u'] = z.u.tolist()
        if not feasible:
            record["outcome"] = "repair_discrete_infeasible"
            return None
        return x

    def search_statistics(self):
        """Count whole-day proposals and first validations, never block feasibility."""
        statistics = {}
        for source in ('sa', 'pool', 'llm', 'llm_rebase'):
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
                new_feasible=sum(r['feasible'] for r in evaluations),
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
        entry = dict(source=source, feasible=r.feasible, dispatch_status=r.status,
                     objective=plan.objective if r.feasible else None,
                     fixed_cost=binary_cost, expected_fuel_cost=plan.objective-binary_cost if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=commitment_hash(plan),
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        if record is not None:
            entry["proposal_id"] = record.get("proposal_id")
            record["incumbent_decrease"] = float(before-plan.objective) if improved and np.isfinite(before) else 0.
        self.cache[key] = entry
        self.diagnostics["evaluations"].append(entry)
        if record is not None:
            record.update(outcome="accepted" if r.feasible else "infeasible", feasible=r.feasible,
                          continuous_dispatch_completed=True, validation_completed=r.feasible)
        if improved:
            self._adopt_incumbent(r, plan, report)
        if source in ("llm", "llm_rebase") and r.feasible:
            self.advisor_pool.append((r.cost, x.copy()))
            self.advisor_pool.sort(key=lambda item: item[0])
            del self.advisor_pool[ADVISOR_POOL_SIZE:]
        self.emit("evaluation")
        return r.feasible

    # 以已校验的可行解替换 incumbent，并保存各场景调度及按场景概率还原的平衡/备用对偶价格。
    def _adopt_incumbent(self, r, plan, report):
        self.best = r
        for name in ("objective", "u", "v", "w", "scenarios"):
            setattr(self.result, name, getattr(plan, name))
        self.result.validation = report
        # 联合 QP 的价格含场景概率，并采用 A*x <= rhs 的符号约定。
        labels = np.asarray(self.full.labels)
        balance = r.prices[labels == "balance"].reshape((-1, len(self.scenarios)))
        reserve = r.prices[labels == "reserve"].reshape((-1, len(self.scenarios)))
        self.dispatches = [NS(**vars(plan.scenarios[k]),
            balance_dual=-balance[:, k] / s.probability,
            reserve_dual=reserve[:, k] / s.probability) for k, s in enumerate(self.scenarios)]

    # 按当前最好方案刷新费用、场景价格和上一轮反馈，返回 (上下文, 基础方案, 调用前快照)。
    # 快照含 u、费用及已评估身份，用于区分本次候选的实测收益和历史缓存结果。
    def upper_context(self):
        phase = perf_counter()
        measurements = self.diagnostics["evaluations"]
        evaluations = [CandidateEvaluation(r["source"], r["objective"] if r["feasible"] else np.inf,
            r["fixed_cost"], r["expected_fuel_cost"] if r["feasible"] else np.inf, r["feasible"])
            for r in measurements]
        context = build_llm_context(self.d, self.scenarios, LLMContextInput(
            self.result, self.dispatches, evaluations,
            [NS(**row) for row in self.diagnostics["iterations"]]))
        context["previous_call_feedback"] = self.feedback
        context['reachability'] = deepcopy(self.diagnostics['reachability'])
        baseline_u = self.result.u.copy()
        baseline = next(r for r in reversed(measurements) if r["commitment_hash"] == commitment_hash(self.result))
        costs = dict(total=baseline["objective"], fixed=baseline["fixed_cost"], expected_fuel=baseline["expected_fuel_cost"])
        prior_ids = {r["commitment_hash"] for r in measurements}
        pool = sorted(((key, row) for key, row in self.cache.items() if row["feasible"]),
                      key=lambda item: item[1]["objective"])[:MAX_CANDIDATES]
        bases = [commitment(self.d, np.frombuffer(key, dtype=np.uint8)) for key, _ in pool]
        self.timings["context"] += perf_counter() - phase
        return context, bases, (baseline_u, costs, prior_ids)

    # 修复并评估上界建议，通过全天验证的方案再投影到所有当前价格块，重新调度生成局部割。
    # 全天割不能直接复制成局部割；最后按调用前快照汇总候选反馈。
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
                        projection = search.evaluate(x[b.cols[b.binary]], "llm")
                        projections.append(dict(block=search.index, completed=projection is not None,
                                                feasible=projection.feasible if projection is not None else None))
                    record["block_projections"] = projections
                    # 没有块搜索时没有局部调度，局部可行性未知，不能按空集记为可行。
                    record["local_dispatch_completed"] = (bool(searches) and len(projections) == len(searches)
                        and all(row["completed"] for row in projections))
                    record["local_feasible"] = (all(row["feasible"] for row in projections)
                        if record["local_dispatch_completed"] else None)
        else:
            for record in info.get("proposals", []):
                record["outcome"] = "deadline"
        rebased = (self.rebase(proposals, baseline_u)
                   if self.remaining() > 0 and not np.array_equal(self.result.u, baseline_u) else [])
        info["rebase"] = rebased
        info["accepted"] = sum(p.get("outcome") == "accepted" for p in info.get("proposals", []))
        self.feedback = summarize(info, baseline_u, costs, self.diagnostics["evaluations"], prior_ids,
                                  shortfall=lambda u: reachable_shortfall(self.d, self.scenarios, u))
        self.feedback["rebase"] = dict(attempts=len(rebased),
            improved=sum(r.get("incumbent_decrease", 0.) > 0 for r in rebased),
            incumbent_decrease=float(sum(r.get("incumbent_decrease", 0.) for r in rebased)))
        info["feedback"] = self.feedback
        self.diagnostics["upper_advice_feedback"] = self.feedback

    # 建议基于请求时的快照生成，返回时 incumbent 可能已被搜索改写。把每个建议相对快照
    # 改动的机组行移植到当前 incumbent：先整体移植，再逐台贪心移植；有改善即成为新的基准。
    def rebase(self, proposals, baseline_u):
        records = []
        for raw, record in proposals:
            raw = np.asarray(raw, float)
            if raw.shape != baseline_u.shape or not np.isfinite(raw).all():
                continue
            raw = np.clip(np.rint(raw), 0, 1)
            units = np.flatnonzero((raw != baseline_u).any(axis=1))
            groups = [units, *([i] for i in units)] if len(units) > 1 else [units]
            for k, group in enumerate(groups if len(units) else []):
                if self.remaining() <= 0:
                    return records
                u = self.result.u.copy(); u[group] = raw[group]
                if np.array_equal(u, self.result.u):
                    continue
                entry = {"proposal_id": f"{record['proposal_id']}:rebase:{k}",
                         "rebased_from": record["proposal_id"], "units": [int(i) + 1 for i in group]}
                self.accept(self.prepare(u, "llm_rebase", entry), "llm_rebase", entry)
                records.append(entry)
        return records


# 一个时间块的原始邻域搜索，整个运行只建立一次。块费用按固定价格定价（只含被松弛的连续耦合行），
# 跨块纯二进制行代入当前 incumbent 作为硬边界行；跨轮保存块内 Benders 割、调度缓存和定价费用最低的块点。
class BlockSearch:
    """Persistent primal neighbourhood search of one time block at fixed prices."""

    # 绑定定价后的块，建立块内硬约束行、跨块边界行和局部诊断。
    def __init__(self, b, o, index, whole):
        self.b, self.o, self.index, self.whole = b, o, index, whole
        # 全天推出的块内固定位，与块内硬约束行一起进入搜索 QUBO；价格固定，硬约束行只需建立一次。
        self.fixed = whole.fixed[b.cols[b.binary]].copy()
        self.hard = hard_rows(b, fixed=self.fixed)
        self.best_local = None
        self.cuts, self.cache, self.history = [], {}, []
        self.cut_totals = dict(optimality=0, feasibility=0)
        # 跨块纯二进制行（relaxed_rows 的补集）：块外列代入当前 incumbent 后成为块内硬约束，只进入搜索 QUBO。
        m = whole.m
        rows = m.A[b.cross].tocsr()
        inside = np.zeros(rows.shape[1], bool); inside[b.cols[b.binary]] = True
        keep = ~relaxed_rows(m, b.cross) & (rows[:, inside].getnnz(axis=1) > 0)
        rows = rows[keep]
        self.boundary = NS(inner=rows[:, b.cols[b.binary]].tocsr(),
                           outer=(rows @ sparse.diags((~inside).astype(float))).tocsr(),
                           rhs=m.rhs[b.cross][keep], eq=m.eq[b.cross][keep])
        self.diagnostic = dict(block=index, hours=(b.times + 1).tolist(), visits=0,
                               sa_calls=0, first_visit_seconds=None, cuts_dropped=0)
        whole.diagnostics["blocks"].append(self.diagnostic)
        self.summarize()

    # 返回跨块边界行在当前 incumbent 下的块内形式：(块内二进制序号, 系数, 右端, 是否等式)。
    def boundary_rows(self):
        ring = self.boundary
        rhs = ring.rhs - ring.outer @ self.whole.best.point
        return [(ring.inner.indices[ring.inner.indptr[k]:ring.inner.indptr[k+1]],
                 ring.inner.data[ring.inner.indptr[k]:ring.inner.indptr[k+1]], rhs[k], ring.eq[k])
                for k in range(len(rhs))]

    # 刷新本块的最好定价费用、累计学到的两类割数、当前保存的割数与缓存规模，供进度与终态诊断读取。
    def summarize(self):
        self.diagnostic.update(
            best_local_cost=self.best_local.cost if self.best_local is not None else None,
            optimality_cuts=self.cut_totals["optimality"], feasibility_cuts=self.cut_totals["feasibility"],
            stored_cuts=len(self.cuts), cache_size=len(self.cache))

    # 收录一条块内割；超过 CUT_LIMIT 时丢弃最早的一条并计数。累计数不因丢弃而减少。
    def add_cut(self, cut):
        self.cuts.append(cut)
        self.cut_totals["optimality" if cut.optimality else "feasibility"] += 1
        if len(self.cuts) > CUT_LIMIT:
            del self.cuts[0]
            self.diagnostic["cuts_dropped"] += 1

    # 交给搜索 QUBO 的割：全部最优性割（QUBO 只取 anchor 处最紧的一条作费用代理），以及 anchor 处
    # 松弛量最小（alpha + beta·anchor 最大）的至多 QUBO_FEASIBILITY_CUTS 条可行割；保持原有顺序。
    def qubo_cuts(self, anchor):
        feasibility = [k for k, c in enumerate(self.cuts) if not c.optimality]
        chosen = set(feasibility)
        if len(feasibility) > QUBO_FEASIBILITY_CUTS:
            values = np.array([self.cuts[k].alpha + self.cuts[k].beta @ anchor for k in feasibility])
            chosen = {feasibility[k] for k in np.argsort(-values, kind="stable")[:QUBO_FEASIBILITY_CUTS]}
        return [c for k, c in enumerate(self.cuts) if c.optimality or k in chosen], len(feasibility), len(chosen)

    # 在固定价格下评估精确二进制块点（块内约束，跨块行松弛），缓存调度并学习块内割。
    # 超时或数值状态不缓存；定价费用只用于块内比较和 QUBO 代理，不能替代全天原始费用。
    def evaluate(self, x, source):
        b, whole = self.b, self.whole
        key = np.asarray(x, np.uint8).tobytes()
        if whole.remaining() <= 0:
            return
        if key in self.cache:
            return self.cache[key]
        phase = perf_counter()
        r = dispatch(b, x, whole.remaining())
        whole.timings["local_dispatch"] += perf_counter() - phase
        if whole.remaining() <= 0:
            return
        if r.feasible:
            residual = b.A @ r.point - b.rhs
            residual[b.eq] = np.abs(residual[b.eq])
            if residual.max(initial=0) > 1e-5:
                raise RuntimeError("Local dispatch validation failed")
        elif r.status not in ("INFEASIBLE", "DISCRETE_INFEASIBLE"):
            # 与全天调度相同：数值状态按未完成处理，不缓存。
            whole.diagnostics["dispatch_failures"] = whole.diagnostics.get("dispatch_failures", 0) + 1
            return
        if whole.remaining() <= 0:
            return
        # 缓存只保留紧凑记录；完整调度结果只为定价费用最低的块点保留。
        record = self.cache[key] = NS(feasible=r.feasible, status=r.status, cost=r.cost)
        if r.cut is not None:
            self.add_cut(r.cut)
        if r.feasible and (self.best_local is None or r.cost < self.best_local.cost):
            self.best_local = r
        whole.diagnostics["local_evaluations"].append(dict(block=self.index, source=source,
            feasible=r.feasible, cost=r.cost if r.feasible else None))
        return record

    # 完成本块的一次访问：先评估种子，再采样一批 SA，评估原始块候选并把修复后的全天方案交给全天校验。
    # 整批候选共享采样前的全天方案快照。
    def step(self, seeds, k):
        b, o, whole = self.b, self.o, self.whole
        if whole.remaining() <= 0:
            return
        self.diagnostic["visits"] += 1
        if self.diagnostic["first_visit_seconds"] is None:
            self.diagnostic["first_visit_seconds"] = perf_counter() - whole.started
        for seed in seeds:
            self.evaluate(seed, "initial")
        if whole.remaining() <= 0:
            return
        if self.best_local is None:
            raise RuntimeError("No feasible block seed")
        baseline_u = whole.result.u.copy()
        anchor_id, anchor_cost = commitment_hash(whole.result), whole.result.objective
        phase = perf_counter()
        columns = b.cols[b.binary]
        boundary = self.boundary_rows()
        anchor = whole.best.point[columns]
        cuts, stored, encoded = self.qubo_cuts(anchor)
        q = search_qubo(self.hard, b, cuts, anchor, boundary, o)
        whole.timings["qubo"] += perf_counter() - phase
        if whole.progress is not None:
            whole.diagnostics["pending_qubo"] = dict(bits=q.Q.shape[0], nonzeros=q.Q.nnz,
                max_coefficient=float(np.abs(q.Q.data).max(initial=0.)))
            whole.emit("before_sampling")
        if whole.remaining() <= 0:
            return
        phase = perf_counter()
        whole.diagnostics["sa_calls"] += 1
        self.diagnostic["sa_calls"] += 1
        # 每条链的种子由配置种子、全局调用序号和链号派生：可复现且互不相同。
        sample_seeds = [int(np.random.SeedSequence([o.seed, whole.diagnostics["sa_calls"], j]).generate_state(1)[0])
                        for j in range(o.sa_reads)]
        proposals, sampling = anneal(q, o, whole.remaining(), sample_seeds)
        whole.timings["annealing"] += perf_counter() - phase
        # 采样诊断只保留标量与哈希；anchor 是本批共享的 incumbent 身份。
        sampling.update(block=self.index, round=k, seed=sample_seeds, boundary_rows=len(boundary),
                        feasibility_cuts_stored=stored, feasibility_cuts_encoded=encoded,
                        anchor=anchor_id, anchor_cost=anchor_cost,
                        raw_candidates=len(proposals), raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals),
                        unique_uvw=len({x.astype(np.uint8).tobytes() for x in proposals}))
        whole.diagnostics["sampling"].append(sampling)
        whole.emit("sampling")
        del q
        if whole.remaining() <= 0:
            return
        # 整批候选都拼接到采样前的 incumbent 快照上，不受本批中途的改进影响。
        for j, x in enumerate(proposals):
            if whole.remaining() <= 0:
                return
            self.evaluate(x, 'sa_raw')  # 原始块候选也参与局部割学习
            block_x = np.zeros(len(b.obj)); block_x[b.binary] = x
            raw = baseline_u.copy(); raw[:, b.times] = block_x[b.u]
            record = {"proposal_id": f"sa:{whole.diagnostics['sa_calls']}:{j}", "block": self.index, "round": k}
            repaired = whole.prepare(raw, "sa", record)
            if repaired is not None and whole.remaining() > 0:
                repaired_eval = self.evaluate(repaired[columns], "sa")
                record.update(local_dispatch_completed=repaired_eval is not None,
                              local_feasible=repaired_eval.feasible if repaired_eval is not None else None)
                whole.accept(repaired, "sa", record)
        if whole.remaining() <= 0:
            return
        self.history.append(dict(block=self.index, round=k, best_local_cost=self.best_local.cost,
                                 optimality_cuts=self.cut_totals["optimality"],
                                 feasibility_cuts=self.cut_totals["feasibility"], sampling=sampling))
        whole.result.history.append(self.history[-1])
        self.summarize()
