"""Round-robin block search with shared whole-day validation and advice."""
# 模块职责：组织全天初始化、固定乘子下的分块 SA 轮询、协调、LLM 建议与独立强化下界认证。
# 全天物理/费用验证通过才更新原问题上界；块价格环境、修复来源及认证证明分别维护。
from dataclasses import dataclass, field
import hashlib
from copy import deepcopy
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from .anneal import anneal
from .benders import coordinate, discrete_feasible, dispatch, master
from .context import CandidateEvaluation, LLMContextInput, build_llm_context
from .upper_advisor import UpperAdvisor, LLMCallFailures, summarize
from .strengthened_certification import StrengthenedCertification
from .model import fingerprint, model, partition
from .optimization import optimize
from .qubo import qubo
from .validation import fixed_cost, capacity, commitment, commitment_hash, pack, repair, unpack, validate


def context_id(b, outer, block):
    digest = hashlib.sha256(np.asarray(b.obj, '<f8').tobytes())
    digest.update(np.asarray([b.constant], '<f8').tobytes())
    digest.update(np.asarray(b.cols, '<i8').tobytes())
    return f'{outer}:{block}:{digest.hexdigest()}'


# 可序列化求解结果：objective 是已验证方案费用，lower_bound 是当前模式认可的全局下界。
# coordination_upper 属于协调对偶模型；状态、原问题 gap 与诊断记录需分别解释。
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

    # 对全天二进制方案去重、做连续调度和独立物理/费用校验，记录实测结果及可用认证割。
    # 只有真实费用更低的可行方案更新 incumbent；命中缓存的重复候选不重复贡献降本。
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
        entry = dict(source=source, feasible=r.feasible, objective=z.objective if r.feasible else None,
                     fixed_cost=fixed, expected_fuel_cost=z.objective-fixed if r.feasible else None,
                     scenario_costs=scenario_costs, commitment_hash=commitment_hash(z),
                     incumbent_before=before, incumbent=min(before, r.cost), improved=improved,
                     elapsed_seconds=perf_counter()-self.started, sa_calls=self.diagnostics["sa_calls"])
        if record is not None:
            entry["proposal_id"] = record.get("proposal_id")
            record["incumbent_decrease"] = float(before-z.objective) if improved else 0.
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
        if source == "llm" and r.feasible:
            self.advisor_pool.append((r.cost, x.copy()))
            self.advisor_pool.sort(key=lambda item: item[0])
            del self.advisor_pool[16:]
        self.emit("evaluation")
        return r.feasible

    # 恢复匹配模型的历史证据、割、调用额度和 SA 配额，并重新调度/校验保存的全天方案。
    # 块缓存与乘子重新构建，Gurobi 内部搜索树及在途工作进程不随检查点恢复。
    def restore(self, state):
        """Revalidate an incumbent and continue this strengthened run's charged budget."""
        saved = deepcopy(state)
        diagnostic = saved["diagnostics"]
        if diagnostic.get("model_fingerprint") != self.diagnostics["model_fingerprint"]:
            raise ValueError("Recovered model fingerprint does not match the current inputs")
        if np.asarray(saved["u"]).shape != (self.d.n_units, self.d.horizon):
            raise ValueError("Recovered commitment dimensions do not match the current case")
        self.certifier.restore(diagnostic)
        self.result.certification_history = saved["certification_history"]
        self.result.llm_history, self.result.llm_calls = saved["llm_history"], saved["llm_calls"]
        for name in ("evaluations", "proposals", "sampling", "rounds", "iterations", "blocks", "local_evaluations"):
            self.diagnostics[name] = diagnostic.get(name, [])
        self.result.history = saved.get('history', [])
        self.result.outer_history = saved.get('outer_history', [])
        self.diagnostics["llm_calls"] = self.result.llm_history
        self.diagnostics["llm_background_seconds"] = sum(row.get("background_seconds", 0.) for row in self.result.llm_history)
        self.resume_cursor = saved["resume_cursor"]
        for name in ("sa_calls", "complete_rounds", "coordination_count"):
            self.diagnostics[name] = diagnostic[name]
        self.timings["prior_execution"] = saved["charged_elapsed_seconds"]
        self.diagnostics['sa_budget_seconds_used'] = diagnostic.get('sa_budget_seconds_used', 0.)
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
            if "u" not in row:
                continue
            x = pack(self.d, np.asarray(row["u"]))
            self.cache[x.astype(np.uint8).tobytes()] = row
            if row["feasible"] and row["source"] == "llm":
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
        info["accepted"] = sum(p.get("outcome") == "accepted" for p in info.get("proposals", []))
        self.feedback = summarize(NS(diagnostics=info, failed=info["status"] != "OK",
                                     message=info.get("message", "")),
                                  baseline_u, costs, self.diagnostics["evaluations"], prior_ids)
        info["feedback"] = self.feedback
        self.diagnostics["upper_advice_feedback"] = self.feedback


# 固定乘子下单个时间块的持久搜索状态，跨轮保存割、调度缓存、最好方案和协调平面。
class BlockSearch:
    """One fixed-multiplier block, retained across round-robin visits."""

    # 绑定块与价格上下文，初始化连续费用基础下界、空割池和局部诊断。
    def __init__(self, b, o, outer, index, whole, planes=None):
        self.b, self.o, self.outer, self.index, self.whole = b, o, outer, index, whole
        self.context = context_id(b, outer, index)
        self.lower = float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant)
        self.upper, self.best = np.inf, None
        self.cuts, self.cache, self.history = [], {}, []
        self.planes = [] if planes is None else planes
        self.diagnostic = dict(outer=outer, block=index, hours=(b.times + 1).tolist(), visits=0,
                               sa_calls=0, first_visit_seconds=None)
        whole.diagnostics["blocks"].append(self.diagnostic)
        self.summarize()

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
            sample_seed = o.seed
            proposals, sampling = anneal(q, o, remaining(), sample_seed)
            whole.timings["annealing"] += perf_counter() - phase
            sampling.update(outer=outer, block=index, inner=inner, round=inner, seed=sample_seed,
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
        whole.timings["relaxation"] += perf_counter() - phase
        if remaining() <= 0:
            return
        result.lower = max(result.lower, bound)
        optimality = sum(c.optimality for c in cuts)
        result.history.append(dict(outer=outer, block=index, inner=inner, lower=result.lower,
            upper=result.upper, optimality_cuts=optimality, feasibility_cuts=len(cuts)-optimality, sampling=sampling))
        whole.result.history.append(result.history[-1])
        self.summarize()


# 主流程：初始化/恢复 → 固定价格轮询所有块 → 轮末建议/认证 → 协调更新价格 → 整理结果。
# 累计历史执行时间计入预算；保留阶段名定位失败，并在最终返回前独立复核全天方案。
def solve(d, scenarios, o, *, resume_state=None, progress=None, started_at=None):
    prior = resume_state["charged_elapsed_seconds"] if resume_state is not None else 0.
    if not np.isfinite(prior) or prior < 0 or prior >= o.time_limit:
        raise ValueError("Invalid or exhausted cumulative resume budget")
    started = (perf_counter() if started_at is None else started_at)-prior; deadline = started + o.time_limit
    deadline -= 5.
    result, whole = Result(), None
    searches = []
    stage = "INITIALIZATION"

    # 在 WholeDay 尚未构造时使用本地 deadline，构造后转用当前工作配额调整后的 deadline。
    def remaining():
        return whole.remaining() if whole is not None else deadline - perf_counter()

    try:
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        full.audit_cuts = bool(getattr(progress, 'audit_cuts', False))
        whole = WholeDay(m, full, o, deadline, result, started, progress)
        phase = perf_counter()
        relaxation = optimize(m, remaining())
        whole.timings["initialization"] += perf_counter() - phase
        if remaining() <= 0:
            raise TimeoutError
        if relaxation.status != "OPTIMAL":
            raise RuntimeError(f"Continuous relaxation failed: {relaxation.status}")
        if hasattr(progress, 'relaxation'):
            progress.relaxation(m, relaxation, perf_counter()-started)
        result.lower_bound = relaxation.lower
        if not np.isfinite(result.lower_bound):
            raise RuntimeError("Continuous relaxation has no finite certified lower bound")
        prices = relaxation.prices
        if resume_state is not None:
            stage = "RESUME"
            whole.restore(resume_state)
        else:
            # 连续解只负责界和初始化；取整方案与全开方案都要经修复、调度、独立校验后比较。
            for u in (np.floor(relaxation.x[m.ix.u] + .5), np.ones((d.n_units, d.horizon))):
                record = {"proposal_id": f"initial:{len(whole.diagnostics['proposals'])}"}
                x = whole.prepare(u, "initial", record)
                whole.accept(x, "initial", record)
        if whole.best is None:
            raise RuntimeError("No validated initial schedule")
        whole.emit("initialization")
        whole.certifier.start()
        if not o.sa_enabled and whole.certifier.converged():
            result.status = "CONVERGED"
        try:
            with whole.sa_work():
                blocks, cross = partition(m, o.block_hours, prices) if o.sa_enabled else ([], np.array([], int))
            planes, previous = [[] for _ in blocks], [[] for _ in blocks]
            first_outer, first_round = whole.resume_cursor["outer"], whole.resume_cursor["round"]
            for outer in (range(first_outer, o.max_outer + 1) if o.sa_enabled and result.status != "CONVERGED" else []):
                if remaining() <= 0:
                    raise TimeoutError
                with whole.sa_work():
                    # 新乘子对应新的块费用与局部割环境，历史二进制种子必须重新调度。
                    blocks, _ = partition(m, o.block_hours, prices)
                    searches = [BlockSearch(b, o, outer, j + 1, whole, planes[j]) for j, b in enumerate(blocks)]
                if whole.advice.state["phase"] == "llm":
                    stage = "ADVISOR_RESUME"
                    whole.advice.run_batch(searches)
                    if whole.certifier.converged():
                        result.status = "CONVERGED"
                        break
                # 每个 inner 是完整遍历轮；尚未闭合的块各最多采样一批，已闭合块跳过采样。
                for inner in range(first_round if outer == first_outer else 1, o.max_inner + 1):
                    before = result.objective
                    sa_before = whole.diagnostics["sa_calls"]
                    before_count, before_calls = len(whole.diagnostics["evaluations"]), result.llm_calls
                    before_opt = sum(c.optimality for search in searches for c in search.cuts)
                    before_feas = sum(not c.optimality for search in searches for c in search.cuts)
                    round_record = dict(outer=outer, round=inner, complete=False, visited_blocks=[],
                                        skipped_closed_blocks=[], started_seconds=perf_counter()-started)
                    whole.diagnostics["rounds"].append(round_record)
                    for j, search in enumerate(searches):
                        whole.certifier.step()
                        if whole.certifier.converged():
                            result.status = 'CONVERGED'
                            break
                        if remaining() <= 0:
                            raise TimeoutError
                        if closed(search.lower, search.upper, o.tolerance):
                            round_record["skipped_closed_blocks"].append(j + 1)
                            continue
                        with whole.sa_work():
                            b = search.b
                            stage = f"BLOCK_{outer}_{j + 1}_ROUND_{inner}"
                            seeds = [whole.best.point[b.cols[b.binary]], *(previous[j] if inner == 1 else []),
                                     *(x[b.cols[b.binary]] for _, x in whole.advisor_pool)]
                            search.step(seeds, inner)
                            if remaining() <= 0:
                                raise TimeoutError
                            round_record["visited_blocks"].append(j + 1)
                            if search.best is not None:
                                raw = result.u.copy(); raw[:, b.times] = search.best.point[b.u]
                                record = {"proposal_id": f"pool:{outer}:{inner}:{j+1}",
                                          "outer": outer, "round": inner, "block": j + 1}
                                x = whole.prepare(raw, "pool", record)
                                whole.accept(x, "pool", record)
                            if o.verbose:
                                print(f"[分块进度] 外层={outer} 内层轮次={inner} 时间块={j + 1}/{len(blocks)} "
                                      f"块内[下界, 上界]=[{search.lower:.6f}, {search.upper:.6f}] "
                                      f"全天可行成本(UB)={result.objective:.6f} "
                                      f"认证下界(LB)={result.lower_bound:.6f}", flush=True)
                            if remaining() <= 0:
                                raise TimeoutError
                    if result.status == 'CONVERGED':
                        round_record.update(ended_seconds=perf_counter()-started, stop_reason='CONVERGED')
                        whole.emit('convergence')
                        break
                    # 完整遍历才更新 SA 轮计数和轮末停滞窗口。
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
                    # 数值认证与上界建议都位于完整 SA 轮之后。
                    stage = "CERTIFICATION"
                    whole.certifier.step()
                    stage = "ADVISOR"
                    whole.advice.after_round(before, sa_before, searches, outer, inner,
                                             not (outer == o.max_outer and inner == o.max_inner))
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
                    if whole.certifier.converged():
                        result.status = "CONVERGED"
                        break
                    if all_closed:
                        break
                if result.status == "CONVERGED":
                    break
                with whole.sa_work():
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
                    whole.emit("coordination")
                    dual_lower = max(relaxation.lower, float(sum(search.lower for search in searches)))
                    result.outer_history[-1]["dual_lower"] = dual_lower
                    result.outer_history[-1]["dual_converged"] = closed(dual_lower, upper, o.tolerance, dual_lower)
                    prices = np.zeros(len(m.rhs)); prices[cross] = values
        except SABudgetExhausted:
            whole.diagnostics['sa_budget_exhausted'] = True
            whole.emit('sa_budget_exhausted')
        if result.status != "CONVERGED":
            # SA 达到迭代或配额上限后，保留已有方案与割，继续使用剩余总预算进行数值认证。
            stage = "CERTIFICATION"
            while remaining() > 0 and not whole.certifier.converged():
                whole.certifier.step()
            if remaining() <= 0:
                raise TimeoutError
            result.status = "CONVERGED"
    except LLMCallFailures as error:
        result.status, result.message = 'PAUSED_LLM_FAILURE', str(error)
    except Exception as error:
        result.status = "TIME_LIMIT" if remaining() <= 0 or isinstance(error, TimeoutError) else f"{stage}_FAILED"
        result.message = f"{type(error).__name__}: {error}"
    finally:
        if whole is not None:
            whole.advice.close()
            try:
                whole.certifier.close()
            except Exception as error:
                whole.diagnostics['proof_shutdown_error'] = type(error).__name__
                if result.status != 'PAUSED_LLM_FAILURE':
                    result.status, result.message = 'CERTIFICATION_FAILED', str(error)
    for search in searches:
        search.summarize()
    if np.isfinite(result.objective):
        result.validation = validate(d, scenarios, result)
        if result.lower_bound > result.objective + 1e-5:
            result.status, result.message = "CERTIFICATION_FAILED", "Lower bound exceeds validated cost"
            result.relative_gap = (result.objective - result.lower_bound) / max(1., abs(result.objective))
        else:
            result.relative_gap = max(0., result.objective - result.lower_bound) / max(1., abs(result.objective))
    result.elapsed_seconds = perf_counter() - started
    if whole is not None:
        whole.timings["other"] = max(0., result.elapsed_seconds - sum(whole.timings.values()))
        result.diagnostics.update(evaluation_count=len(whole.cache), llm_call_count=result.llm_calls,
                                  time_limit_overshoot=max(0., result.elapsed_seconds-o.time_limit),
                                  resume_cursor=whole.resume_cursor)
        whole.emit("final")
    if o.verbose:
        print(f"[主算法结束] 状态={result.status} 可行成本(UB)={result.objective:.6f}, "
              f"认证下界(LB)={result.lower_bound:.6f}, "
              f"认证间隙={100 * result.relative_gap:.4f}%, 耗时={result.elapsed_seconds:.2f}秒", flush=True)
    return result
