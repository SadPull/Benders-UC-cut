"""Time-block SA and shared whole-day candidate evaluation."""
# 分块搜索只产生候选；全天修复、调度和独立校验通过后才能更新 UB。
import hashlib
from copy import deepcopy
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
from scipy import sparse

from .anneal import anneal
from .benders import discrete_feasible, dispatch, master
from .context import CandidateEvaluation, LLMContextInput, build_llm_context
from .upper_advisor import UpperAdvisor, summarize
from .strengthened_certification import StrengthenedCertification
from .model import fingerprint
from .optimization import optimize
from .qubo import search_qubo, fixed_binary
from .validation import fixed_cost, capacity, commitment, commitment_hash, pack, repair, unpack, validate, transition_allowed, transition_violations


def context_id(b, outer, block):
    digest = hashlib.sha256(np.asarray(b.obj, '<f8').tobytes())
    digest.update(np.asarray([b.constant], '<f8').tobytes())
    digest.update(np.asarray(b.cols, '<i8').tobytes())
    return f'{outer}:{block}:{digest.hexdigest()}'


# 用固定绝对余量 1e-5 加相对容差判断有限上下界是否闭合，同时拒绝明显交叉的上下界。
# reference 可指定相对容差的计价基准，默认使用 upper。
def closed(lower, upper, tolerance, reference=None):
    reference = upper if reference is None else reference
    return (np.isfinite(lower) and np.isfinite(upper) and upper - lower >= -1e-5
            and upper - lower <= 1e-5 + tolerance * max(1, abs(reference)))


# 辅助 SA 工作配额耗尽的内部信号；与整个求解时间耗尽区分，以便继续数值认证。
class SABudgetExhausted(Exception):
    pass


# 各块共用的全天方案、调度缓存、数值反馈、预算和认证/LLM 管理状态。
class WholeDay:
    """Shared incumbent, measured feedback and deadline across all time blocks."""

    # 绑定输入模型与全局截止时间，建立按来源统计的诊断和计时器，建立强化认证器。
    def __init__(self, m, full, o, deadline, result, started, progress=None):
        self.m, self.full, self.o, self.deadline = m, full, o, deadline
        self.d, self.scenarios, self.result, self.started = m.d, m.scenarios, result, started
        self.best, self.dispatches = None, []
        self.cache, self.advisor_pool = {}, []
        self.feedback = None
        self.diagnostics = result.diagnostics
        self.diagnostics.update(sa_calls=0, evaluations=[], proposals=[], sampling=[], iterations=[],
                                local_evaluations=[], llm_calls=[], llm_background_seconds=0., blocks=[], rounds=[],
                                complete_rounds=0, coordination_count=0, timings={name: 0. for name in (
                                    "initialization", "relaxation", "qubo", "annealing", "repair",
                                    "local_dispatch", "subproblems", "validation", "context", "llm", "coordination")})
        self.timings = self.diagnostics["timings"]
        self.progress = progress
        self.certifier = StrengthenedCertification(self)
        self.diagnostics["bound_mode"] = 'strengthened'
        self.diagnostics["model_fingerprint"] = fingerprint(m)
        self.resume_cursor = dict(outer=1, round=1)
        self.advice = UpperAdvisor(self)
        self.sa_started = None
        self.diagnostics['sa_budget_seconds_used'] = 0.
        self.diagnostics.update(block_sa_calls=0, search_mode='time_blocks')
        phase = perf_counter()
        reachable, _ = master(full, [], search_reachability=True)
        self.fixed = fixed_binary(reachable)[:len(full.binary)]
        start, stop = transition_allowed(self.d)
        fixed_u = self.fixed[full.u]
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
    # 退出时恢复全局 deadline；配额耗尽转为 SABudgetExhausted，而不是结束后续认证。
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
                           mip_focus=1, solution_limit=1)
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
        if source == 'llm' and np.isin(raw, [0, 1]).all():
            record['raw_constraint_violations'] = transition_violations(self.d, raw)
        raw_x = pack(self.d, raw)
        record["raw_binary"] = bool(np.isin(raw, [0, 1]).all())
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
        record.update(commitment_id=commitment_hash(z), repaired_u=z.u.tolist(),
                      hamming_u=int(np.count_nonzero(z.u != raw)), repaired_discrete_feasible=feasible,
                      duplicate=x.astype(np.uint8).tobytes() in self.cache, outcome="repaired")
        if not feasible:
            record["outcome"] = "repair_discrete_infeasible"
            return None
        return x

    def search_statistics(self):
        """Count whole-day proposals and first validations, never block feasibility."""
        statistics = {}
        fixed_u = self.fixed[self.full.u]
        known = np.isfinite(fixed_u)
        for source in ('sa', 'pool', 'llm', 'llm_rebase'):
            proposals = [r for r in self.diagnostics['proposals'] if r['origin'] == source]
            evaluations = [r for r in self.diagnostics['evaluations'] if r['source'] == source]
            raw = {hashlib.sha256(np.asarray(r['raw_u'], '<f8').tobytes()).digest() for r in proposals}
            repaired = {r['commitment_id'] for r in proposals if r.get('repaired_discrete_feasible')}
            outcomes = {}
            for r in proposals:
                outcome = r['outcome']; outcomes[outcome] = outcomes.get(outcome, 0)+1
            statistics[source] = dict(proposals=len(proposals), raw_unique=len(raw), repaired_unique=len(repaired),
                raw_duplicate_rate=1-len(raw)/len(proposals) if proposals else 0.,
                repaired_duplicate_rate=1-len(repaired)/sum(bool(r.get('repaired_discrete_feasible')) for r in proposals)
                    if repaired else 0.,
                raw_discrete_violations=sum(r.get('raw_discrete_feasible') is False for r in proposals),
                raw_fixed_bit_violations=sum(int(np.count_nonzero(np.asarray(r['raw_u'])[known] != fixed_u[known]))
                                            for r in proposals),
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
        key = x.astype(np.uint8).tobytes()
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
        improved = r.feasible and z.objective < before
        entry = dict(source=source, feasible=r.feasible, dispatch_status=r.status,
                     objective=z.objective if r.feasible else None,
                     fixed_cost=fixed, expected_fuel_cost=z.objective-fixed if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=commitment_hash(z),
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        if record is not None:
            entry["proposal_id"] = record.get("proposal_id")
            record["incumbent_decrease"] = float(before-z.objective) if improved and np.isfinite(before) else 0.
        self.cache[key] = entry
        if r.feasible:
            entry["hourly_recourse_costs"] = [float(
                self.full.obj[indices] @ r.point[indices]
                + self.full.q[indices] @ r.point[indices]**2)
                for hour in self.full.times
                for indices in [self.full.continuous[self.full.time[self.full.continuous] == hour]]]
        entry["u"] = z.u.tolist()
        self.diagnostics["evaluations"].append(entry)
        if record is not None:
            record.update(outcome="accepted" if r.feasible else "infeasible", feasible=r.feasible,
                          continuous_dispatch_completed=True, validation_completed=r.feasible)
        if improved:
            if not np.isfinite(before):
                self.diagnostics["initial_upper"] = z.objective
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
        if source in ("llm", "llm_rebase") and r.feasible:
            self.advisor_pool.append((r.cost, x.copy()))
            self.advisor_pool.sort(key=lambda item: item[0])
            del self.advisor_pool[16:]
        self.emit("evaluation")
        return r.feasible

    # 恢复匹配模型的历史证据、调用额度和 SA 配额，并重新调度/校验保存的全天方案。
    # 块缓存与乘子重新构建，Gurobi 内部搜索树及在途工作进程不随检查点恢复。
    def restore(self, state):
        """Revalidate an incumbent and continue this strengthened run's charged budget."""
        saved = deepcopy(state)
        diagnostic = saved["diagnostics"]
        if diagnostic.get("model_fingerprint") != self.diagnostics["model_fingerprint"]:
            raise ValueError("Recovered model fingerprint does not match the current inputs")
        if np.asarray(saved["u"]).shape != (self.d.n_units, self.d.horizon):
            raise ValueError("Recovered commitment dimensions do not match the current case")
        if diagnostic.get('search_mode') != 'time_blocks':
            raise ValueError('Checkpoint must use the current time-block format')
        if (diagnostic.get('joint_sa_calls', 0) or diagnostic.get('joint_batches')
                or any(row.get('kind') == 'joint' for row in diagnostic.get('sampling', []))):
            raise ValueError('Joint-SA checkpoints are unsupported; start a new time-block run')
        self.certifier.restore(diagnostic)
        self.result.certification_history = saved["certification_history"]
        self.result.llm_history, self.result.llm_calls = saved["llm_history"], saved["llm_calls"]
        for name in ("evaluations", "proposals", "sampling", "rounds", "iterations", "blocks", "local_evaluations"):
            self.diagnostics[name] = diagnostic[name]
        self.result.history = saved['history']
        self.result.outer_history = saved['outer_history']
        self.diagnostics["llm_calls"] = self.result.llm_history
        self.diagnostics["llm_background_seconds"] = diagnostic["llm_background_seconds"]
        self.resume_cursor = saved["resume_cursor"]
        for name in ("sa_calls", "complete_rounds", "coordination_count"):
            self.diagnostics[name] = diagnostic[name]
        self.diagnostics['block_sa_calls'] = diagnostic['block_sa_calls']
        self.timings["prior_execution"] = saved["charged_elapsed_seconds"]
        self.diagnostics['sa_budget_seconds_used'] = diagnostic['sa_budget_seconds_used']
        self.diagnostics["resume"] = saved["resume"]
        x = pack(self.d, np.asarray(saved["u"]))
        phase = perf_counter()
        checked = dispatch(self.full, x, self.remaining(), certified=True)
        self.timings["subproblems"] += perf_counter()-phase
        if self.remaining() <= 0 or checked.status == 'TIME_LIMIT':
            raise TimeoutError
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
        reserve = checked.prices[labels == "reserve"].reshape((-1, len(self.scenarios)))
        self.dispatches = [NS(**vars(point.scenarios[k]), balance_dual=-balance[:, k]/s.probability,
                              reserve_dual=reserve[:, k]/s.probability)
                           for k, s in enumerate(self.scenarios)]
        self.certifier.update_bound(saved["lower_bound"])
        for row in self.diagnostics["evaluations"]:
            x = pack(self.d, np.asarray(row["u"]))
            self.cache[x.astype(np.uint8).tobytes()] = row
            if row["feasible"] and row["source"] in ("llm", "llm_rebase"):
                self.advisor_pool.append((row["objective"], x.copy()))
        self.advisor_pool.sort(key=lambda item: item[0])
        del self.advisor_pool[16:]
        self.advice.restore(diagnostic)

    # 按当前最好方案刷新费用、场景价格和上一轮反馈，返回 (上下文, 基础方案, 调用前快照)。
    # 快照含 u、费用及已评估身份，用于区分本次候选的实测收益和历史缓存结果。
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
        context['reachability'] = deepcopy(self.diagnostics['reachability'])
        baseline_u = self.result.u.copy()
        baseline = next(r for r in reversed(measurements) if r["commitment_hash"] == commitment_hash(self.result))
        costs = dict(total=baseline["objective"], fixed=baseline["fixed_cost"], expected_fuel=baseline["expected_fuel_cost"])
        prior_ids = {r["commitment_hash"] for r in measurements}
        pool = sorted(((key, row) for key, row in self.cache.items() if row["feasible"]),
                      key=lambda item: item[1]["objective"])[:16]
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
        rebased = (self.rebase(proposals, baseline_u)
                   if self.remaining() > 0 and not np.array_equal(self.result.u, baseline_u) else [])
        info["rebase"] = rebased
        info["accepted"] = sum(p.get("outcome") == "accepted" for p in info.get("proposals", []))
        self.feedback = summarize(NS(diagnostics=info, failed=info["status"] != "OK",
                                     message=info.get("message", "")),
                                  baseline_u, costs, self.diagnostics["evaluations"], prior_ids)
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


# 固定乘子下单个时间块的持久搜索状态，跨轮保存割、调度缓存、最好方案和协调平面。
class BlockSearch:
    """One fixed-multiplier block, retained across round-robin visits."""

    # 绑定块与价格上下文，初始化连续费用基础下界、空割池和局部诊断。
    def __init__(self, b, o, outer, index, whole, planes=None):
        self.b, self.o, self.outer, self.index, self.whole = b, o, outer, index, whole
        b.search_fixed = whole.fixed[b.cols[b.binary]].copy()
        self.context = context_id(b, outer, index)
        self.lower = float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant)
        self.upper, self.best = np.inf, None
        self.cuts, self.cache, self.history = [], {}, []
        self.planes = [] if planes is None else planes
        # 跨块纯二进制行：块外列代入当前 incumbent 后成为块内硬约束，只进入搜索 QUBO。
        m = whole.m
        rows = m.A[b.cross].tocsr()
        continuous = np.ones(rows.shape[1], bool); continuous[m.binary] = False
        inside = np.zeros(rows.shape[1], bool); inside[b.cols[b.binary]] = True
        keep = (rows[:, continuous].getnnz(axis=1) == 0) & (rows[:, inside].getnnz(axis=1) > 0)
        rows = rows[keep]
        self.boundary = NS(inner=rows[:, b.cols[b.binary]].tocsr(),
                           outer=(rows @ sparse.diags((~inside).astype(float))).tocsr(),
                           rhs=m.rhs[b.cross][keep], eq=m.eq[b.cross][keep])
        self.diagnostic = dict(outer=outer, block=index, hours=(b.times + 1).tolist(), visits=0,
                               sa_calls=0, first_visit_seconds=None)
        whole.diagnostics["blocks"].append(self.diagnostic)
        self.summarize()

    # 返回跨块边界行在当前 incumbent 下的块内形式：(块内二进制序号, 系数, 右端, 是否等式)。
    def boundary_rows(self):
        ring = self.boundary
        rhs = ring.rhs - ring.outer @ self.whole.best.point
        return [(ring.inner.indices[ring.inner.indptr[k]:ring.inner.indptr[k+1]],
                 ring.inner.data[ring.inner.indptr[k]:ring.inner.indptr[k+1]], rhs[k], ring.eq[k])
                for k in range(len(rhs))]

    # 刷新本块的局部上下界、gap、收敛标记及两类割数量，供进度与终态诊断读取。
    def summarize(self):
        self.diagnostic.update(lower=self.lower, upper=self.upper,
            relative_gap=(max(0., self.upper-self.lower)/max(1., abs(self.upper))
                          if np.isfinite(self.upper) else None),
            converged=closed(self.lower, self.upper, self.o.tolerance),
            optimality_cuts=sum(c.optimality for c in self.cuts),
            feasibility_cuts=sum(not c.optimality for c in self.cuts))

    # 返回至多十六个费用较好的历史可行二进制方案，用于重新调度；不是额外注入 SA 样本。
    @property
    def seeds(self):
        pool = sorted((r for r in self.cache.values() if r.feasible), key=lambda r: r.cost)[:16]
        return [r.point[self.b.binary].astype(np.uint8) for r in pool]

    # 在当前块价格环境中评估精确二进制点，缓存调度、更新局部割和可行协调平面。
    # 超时结果不缓存；局部费用含当前拉格朗日价格，不能直接替代全天原始费用。
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
        r = (dispatch(b, x, remaining()))
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
        whole.diagnostics["local_evaluations"].append(dict(outer=self.outer, block=self.index, source=source,
            feasible=r.feasible, cost=r.cost if r.feasible else None, context=self.context))
        return r

    # 完成本块的一次访问：重调度种子、求 LP 下界，未闭合时最多采样一批 SA，再评估并刷新下界。
    # 整批候选共享采样前的全天方案快照。
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
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            return
        result.lower = max(result.lower, bound)
        sampling = {"bits": 0, "candidates": 0, "best_energy": None, "seconds": 0.}
        if not closed(result.lower, result.upper, o.tolerance):
            if result.best is None:
                raise RuntimeError("No feasible block seed")
            baseline_u = whole.result.u.copy()
            anchor_id, anchor_cost = commitment_hash(whole.result), whole.result.objective
            phase = perf_counter()
            local = b.cols[b.binary]
            anchors = [whole.best.point[local]]
            if whole.advisor_pool and not np.array_equal(whole.advisor_pool[0][1][local], anchors[0]):
                anchors.append(whole.advisor_pool[0][1][local].astype(float))
            boundary = self.boundary_rows()
            q = search_qubo(m, b, cuts, anchors, boundary, o)
            whole.timings["qubo"] += perf_counter() - phase
            if whole.progress is not None:
                whole.diagnostics["pending_qubo"] = dict(bits=q.Q.shape[0], nonzeros=q.Q.nnz,
                    max_coefficient=float(abs(q.Q).max()), dense_working_bytes=32*(q.Q.shape[0]+1)**2)
                whole.emit("before_sampling")
            if remaining() <= 0:
                return
            phase = perf_counter()
            whole.diagnostics["sa_calls"] += 1
            whole.diagnostics["block_sa_calls"] += 1
            self.diagnostic["sa_calls"] += 1
            # 每条链的种子由配置种子、全局调用序号和链号派生：可复现且互不相同。
            sample_seeds = [int(np.random.SeedSequence([o.seed, whole.diagnostics["sa_calls"], k]).generate_state(1)[0])
                            for k in range(o.sa_reads)]
            proposals, sampling = anneal(q, o, remaining(), sample_seeds)
            whole.timings["annealing"] += perf_counter() - phase
            sampling.update(outer=outer, block=index, inner=inner, round=inner, seed=sample_seeds,
                            anchors=len(anchors), boundary_rows=len(boundary),
                            kind='block', anchor=anchor_id, anchor_cost=anchor_cost,
                            raw_candidates=len(proposals), raw_discrete_feasible=sum(discrete_feasible(b, x) for x in proposals),
                            unique_uvw=len({x.astype(np.uint8).tobytes() for x in proposals}))
            sampling.update(raw_binary_candidates=[x.tolist() for x in proposals],
                            baseline_u=baseline_u.tolist(), free_indices=q.free.tolist(),
                            fixed_indices=np.flatnonzero(~np.isnan(q.fixed)).tolist(),
                            fixed_values=q.fixed[~np.isnan(q.fixed)].tolist())
            whole.diagnostics["sampling"].append(sampling)
            whole.emit("sampling")
            del q
            if remaining() <= 0:
                return
            # All samples use the same incumbent snapshot, before this batch's improvements.
            for k, x in enumerate(proposals):
                if remaining() <= 0:
                    return
                raw_evaluation = evaluate(x, 'sa_raw')
                local = np.zeros(len(b.obj)); local[b.binary] = x
                raw = baseline_u.copy(); raw[:, b.times] = local[b.u]
                record = {"proposal_id": f"sa:{whole.diagnostics['sa_calls']}:{k}", "outer": outer, "block": index, "round": inner}
                record['anchor'] = anchor_id
                record['raw_local_dispatch_completed'] = raw_evaluation is not None
                record['raw_local_feasible'] = raw_evaluation.feasible if raw_evaluation is not None else None
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
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            return
        result.lower = max(result.lower, bound)
        optimality = sum(c.optimality for c in cuts)
        result.history.append(dict(outer=outer, block=index, inner=inner, lower=result.lower,
            upper=result.upper, optimality_cuts=optimality, feasibility_cuts=len(cuts)-optimality, sampling=sampling))
        whole.result.history.append(result.history[-1])
        self.summarize()
