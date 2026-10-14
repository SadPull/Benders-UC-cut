"""Persistent whole-day integer Benders bounds, separate from block multipliers."""
# 模块职责：维护可选全天整数 Benders 主问题及全天/小时/场景认证割，持续更新全局下界。
# 本模块仅用于 integer_benders 模式；主问题 ObjBound 可作下界，整数候选 ObjVal 仅用于探测。
import hashlib
from types import SimpleNamespace as NS
from time import perf_counter

import gurobipy as gp
import numpy as np

from . import bound_advisor
from .benders import master
from .hourly_bounds import HourlyBounds
from .scenario_bounds import ScenarioBounds
from .validation import commitment, commitment_hash, pack


# 与块乘子分离的全天认证状态：保存割、已同步位置、主问题及认证窗口。
class Certification:
    # 绑定 WholeDay 并建立认证割池和小时子问题；改进模式另启用场景割及自适应窗口。
    # 此时只准备数据，实际 Gurobi 模型在首次认证时创建。
    def __init__(self, whole):
        self.whole, self.b, self.o, self.result = whole, whole.full, whole.o, whole.result
        self.cuts, self.identities, self.synced = [], set(), 0
        self.g, self.env, self.variables, self.master_data = None, None, None, None
        self.warm_key, self.last_master, self.last_probe = None, None, None
        self.hourly = HourlyBounds(self.b)
        self.hourly_queue, self.hourly_variables = [], None
        self.hourly_synced = 0
        self.scenario = ScenarioBounds(self.b, whole.m) if self.o.rts24_improvements else None
        self.scenario_variables, self.scenario_synced = None, 0
        self.window_index, self.window_stalls = 0, 0
        self.callback_error = None
        self.cover_cuts = []
        whole.diagnostics.update(certificate_cuts=[], hourly_certificate_cuts=self.hourly.cuts,
                                 bound_probes=[], llm_background_seconds=0.)
        if self.scenario is not None:
            whole.diagnostics.update(scenario_certificate_cuts=self.scenario.cuts, native_bounds=[])
        whole.timings["certification"] = 0.

    # 接收调用方已独立回放且绑定模型指纹、来源哈希的认证证据，更新全局下界。
    # 允许继承的加强割仅限全局容量覆盖割；继承证据与实际连续松弛来源分开记录。
    def inherit(self, evidence):
        """Import a separately replayed global floor; never treat it as an LP dual."""
        from .model import fingerprint
        from .lower_bound import verify_cut
        if (evidence.get('model_fingerprint') != fingerprint(self.whole.m)
                or not evidence.get('source_sha256') or evidence.get('verified') is not True):
            raise ValueError('Inherited bound lacks verified model-bound provenance')
        floor = float(evidence['lower_bound'])
        if not np.isfinite(floor) or floor > self.result.objective+1e-5:
            raise ValueError('Invalid inherited certified lower bound')
        covers = evidence.get('cover_cuts', [])
        for cut in covers:
            if cut.get('kind') != 'cover':
                raise ValueError('Only global capacity covers may be inherited')
            verify_cut(self.whole.m, cut)
        self.cover_cuts = list(covers)
        self.whole.diagnostics['inherited_evidence'] = evidence
        self.update_bound(floor)

    # 从经调用方验证的恢复记录重建全天/小时/场景割与去重集合。
    # 恢复的是数值约束数据，不是 Gurobi 内部搜索树。
    def restore_cuts(self, diagnostic):
        self.whole.diagnostics["certificate_cuts"] = diagnostic["certificate_cuts"]
        self.cuts = [NS(alpha=c["alpha"], beta=np.asarray(c["beta"]),
                        optimality=c["optimality"], certified=True)
                     for c in diagnostic["certificate_cuts"]]
        self.identities = {c["cut_id"] for c in diagnostic["certificate_cuts"]}
        self.hourly.cuts.extend(diagnostic["hourly_certificate_cuts"])
        self.hourly.identities = {c["cut_id"] for c in self.hourly.cuts}
        if self.scenario is not None:
            self.scenario.cuts.extend(diagnostic['scenario_certificate_cuts'])
            self.scenario.identities = {c['cut_id'] for c in self.scenario.cuts}

    # 接收全天连续调度生成的认证割，去重后保留来源、锚点及对当前主问题的违反程度。
    # 可行调度还提供真实小时费用并排入小时割队列，供后续同步处理。
    def observe(self, x, evaluation, source):
        cut = evaluation.cut
        if cut is None:
            return
        if not cut.certified:
            raise RuntimeError("Uncertified cut offered to the whole-day master")
        if evaluation.feasible:
            evaluation.hourly_recourse = self.hourly.recourse(evaluation.point)
            self.hourly_queue.append((x.copy(), source, evaluation.hourly_recourse))
            if self.scenario is not None:
                self.scenario.observe(x, evaluation, source, perf_counter()-self.whole.started)
        identity = hashlib.sha256(np.r_[cut.optimality, cut.alpha, cut.beta].astype("<f8").tobytes()).hexdigest()
        if identity in self.identities:
            return
        self.identities.add(identity)
        self.cuts.append(cut)
        violation = None
        if self.last_master is not None:
            point = self.last_master["x"]
            violation = float(cut.alpha + cut.beta @ point)
            if cut.optimality:
                violation += float(self.b.obj[self.b.binary] @ point + self.b.constant
                                   - self.last_master["predicted_cost"])
        self.whole.diagnostics["certificate_cuts"].append(dict(
            cut_id=identity, source=source, optimality=cut.optimality, alpha=float(cut.alpha),
            beta=cut.beta.tolist(), anchor_hash=commitment_hash(commitment(self.whole.d, x)),
            master_violation=violation, lower_at_creation=self.result.lower_bound,
            elapsed_seconds=perf_counter()-self.whole.started))
        if hasattr(cut, "proof"):
            self.whole.diagnostics["certificate_cuts"][-1]["proof"] = cut.proof

    # 创建单线程整数主问题，将固定费用与小时/场景补偿费用下估计连接到总费用 theta。
    # 主问题包含全天离散约束和已有覆盖割，不包含完整 SUC 的连续出力决策。
    def _initialize(self):
        m, _ = master(self.b, [])
        self.master_data = m
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.g = gp.Model(env=self.env)
        for name, value in dict(Threads=1, Seed=self.o.seed, MIPGap=min(1e-6, self.o.tolerance/10),
                                FeasibilityTol=1e-8, IntFeasTol=1e-9, OptimalityTol=1e-8).items():
            self.g.setParam(name, value)
        kinds = np.full(len(m.obj), gp.GRB.CONTINUOUS)
        kinds[m.binary] = gp.GRB.BINARY
        self.variables = self.g.addMVar(len(m.obj), lb=m.lb, ub=m.ub, vtype=kinds)
        self.g.setObjective(m.obj @ self.variables)
        self.g.addMConstr(m.A, self.variables, np.where(m.eq, "=", "<"), m.rhs)
        mapping = {int(column): j for j, column in enumerate(self.b.binary)}
        for cut in self.cover_cuts:
            self.g.addConstr(self.variables[[mapping[i] for i in cut['indices']]].sum() >= 1.)
        lower, upper = [], []
        for part in self.hourly.parts:
            b, Y = part.b, part.b.continuous
            lower.append(np.minimum(b.obj[Y]*b.lb[Y], b.obj[Y]*b.ub[Y]).sum())
            upper.append(np.sum(np.maximum(b.obj[Y]*b.lb[Y], b.obj[Y]*b.ub[Y]) + b.q[Y]*b.ub[Y]**2))
        self.hourly_variables = self.g.addMVar(len(lower), lb=lower, ub=upper)
        self.g.addConstr(self.hourly_variables.sum() + self.b.obj[self.b.binary] @ self.variables[:m.nx]
                         <= self.variables[-1])
        if self.scenario is not None:
            lower, upper = [], []
            for part in self.scenario.parts:
                b, Y = part.b, part.b.continuous
                lower.append(np.minimum(b.obj[Y]*b.lb[Y], b.obj[Y]*b.ub[Y]).sum())
                upper.append(np.sum(np.maximum(b.obj[Y]*b.lb[Y], b.obj[Y]*b.ub[Y])+b.q[Y]*b.ub[Y]**2))
            self.scenario_variables = self.g.addMVar(len(lower), lb=lower, ub=upper)
            self.g.addConstr(self.scenario_variables.sum()+self.b.obj[self.b.binary] @ self.variables[:m.nx]
                             <= self.variables[-1])
            self.g.Params.PoolSolutions = 10

    # 把尚未处理的小时探测和新增各类割同步到主问题，并用已验证最好方案设置 MIP Start。
    # 模型对象持续存在，但新增约束后的内部搜索状态不能当作可恢复搜索树来承诺。
    def _sync(self):
        m = self.master_data
        for x, source, actual in self.hourly_queue:
            for cut in self.hourly.observe(x, source, actual, self.whole.remaining):
                indices = np.asarray(cut["binary_indices"])
                cut["elapsed_seconds"] = perf_counter()-self.whole.started
                cut["master_violation"] = (float(cut["alpha"] + np.asarray(cut["beta"]) @ self.last_master["x"][indices]
                    - self.last_master["hourly_recourse"][cut["hour"]-1]) if self.last_master is not None else None)
        self.hourly_queue.clear()
        for cut in self.hourly.cuts[self.hourly_synced:]:
            self.g.addConstr(np.asarray(cut["beta"]) @ self.variables[cut["binary_indices"]]
                             <= self.hourly_variables[cut["hour"]-1] - cut["alpha"])
        self.hourly_synced = len(self.hourly.cuts)
        for cut in self.cuts[self.synced:]:
            if cut.optimality:
                self.g.addConstr((cut.beta + self.b.obj[self.b.binary]) @ self.variables[:m.nx]
                                 - self.variables[-1] <= -cut.alpha)
            else:
                self.g.addConstr(cut.beta @ self.variables[:m.nx] <= -cut.alpha)
        self.synced = len(self.cuts)
        if self.scenario is not None:
            for cut in self.scenario.cuts[self.scenario_synced:]:
                self.g.addConstr(np.asarray(cut['beta']) @ self.variables[:m.nx]
                                 <= self.scenario_variables[cut['scenario']]-cut['alpha'])
            self.scenario_synced = len(self.scenario.cuts)
            self.variables[-1].LB = max(m.lb[-1], self.result.lower_bound-self.b.constant)
        point = self.whole.best.point[self.b.binary]
        key = point.astype(np.uint8).tobytes()
        if key != self.warm_key:
            start = np.zeros(len(m.obj))
            start[:m.nx] = point
            for z, v, w in m.and_pairs:
                start[z] = start[v] * start[w]
            start[-1] = self.result.objective - self.b.constant
            self.variables.Start = start
            self.hourly_variables.Start = self.hourly.recourse(self.whole.best.point)
            if self.scenario is not None:
                self.scenario_variables.Start = self.scenario.recourse(self.whole.best.point)
            self.warm_key = key

    # 取有限认证下界与历史值的最大值；超过已验证上界时立即报错。
    def update_bound(self, lower):
        if np.isfinite(lower):
            candidate = max(self.result.lower_bound, float(lower))
            if candidate > self.result.objective + 1e-5:
                raise RuntimeError("Certified lower bound exceeds validated upper bound")
            self.result.lower_bound = candidate

    # 只有全天物理验证通过，且 UB-LB 不超过 1e-5 + tolerance*max(1,|UB|)，才判定原问题收敛。
    def converged(self):
        lower, upper = self.result.lower_bound, self.result.objective
        if lower > upper + 1e-5:
            raise RuntimeError("Certified lower bound exceeds validated upper bound")
        return (np.isfinite(lower) and np.isfinite(upper) and self.result.validation is not None
                and self.result.validation["feasible"]
                and upper-lower <= 1e-5 + self.o.tolerance*max(1., abs(upper)))

    # 通过 WholeDay 评估未经启发式修复的二进制探测点，记录真实费用、低估量和新增割。
    # 可行探测可改善上界，不可行探测可产生可行性割；新增割本身不等于下界已提高。
    def _probe(self, x, source, record):
        before = len(self.cuts)
        self.whole.diagnostics["bound_probes"].append(record)
        self.whole.diagnostics["proposals"].append(record)
        record.update(source=source, raw_u=commitment(self.whole.d, x).u.tolist(),
                      lower_before=self.result.lower_bound)
        accepted = self.whole.accept(x, source, record)
        measured = self.whole.cache.get(x.astype(np.uint8).tobytes())
        record.update(cut_added=len(self.cuts) > before, feasible=accepted,
                      actual_cost=measured["objective"] if measured else None,
                      hourly_actual_recourse=measured.get("hourly_recourse_costs") if measured else None,
                      elapsed_seconds=perf_counter()-self.whole.started)
        if record.get("predicted_cost") is not None and record["actual_cost"] is not None:
            record["underestimation"] = record["actual_cost"] - record["predicted_cost"]
        if self.o.rts24_improvements:
            self.whole.emit('probe_complete')
        return record

    # 验证 LLM 下界候选的形状、0/1 和离散约束，再按已有割估算费用并执行真实探测。
    # 本通道不调用启发式修复，避免改变用于发现低估区域的原始探测点。
    def probe_advice(self, raw, record):
        raw = np.asarray(raw, float)
        if raw.shape != (self.whole.d.n_units, self.whole.d.horizon) or not np.isin(raw, [0, 1]).all():
            record.update(outcome="invalid_matrix", cut_added=False)
            return record
        x = pack(self.whole.d, raw)
        hard = self.b.A[:, self.b.continuous].getnnz(axis=1) == 0
        residual = self.b.A[hard][:, self.b.binary] @ x - self.b.rhs[hard]
        residual[self.b.eq[hard]] = np.abs(residual[self.b.eq[hard]])
        if residual.max(initial=0) > 1e-7:
            record.update(outcome="discrete_infeasible", cut_added=False)
            return record
        Y = self.b.continuous
        recourse = max([float(np.minimum(self.b.obj[Y]*self.b.lb[Y], self.b.obj[Y]*self.b.ub[Y]).sum()),
                        self.hourly.lower(x),
                        self.scenario.lower(x) if self.scenario is not None else -np.inf,
                        *(float(c.alpha + c.beta @ x) for c in self.cuts if c.optimality)])
        record["predicted_cost"] = float(recourse + self.b.obj[self.b.binary] @ x + self.b.constant)
        return self._probe(x, "llm_bound", record)

    # 将当前主问题方案、上下界、小时预测及近期割/探测反馈打包为下界通道请求体。
    def advice_body(self, iteration, stalled=0):
        cut_feedback = [{key: row[key] for key in ("cut_id", "source", "optimality", "master_violation")}
                        for row in self.whole.diagnostics["certificate_cuts"][-12:]]
        context = dict(lower_bound=self.result.lower_bound, upper_bound=self.result.objective,
            relative_gap=(self.result.objective-self.result.lower_bound)/max(1., abs(self.result.objective)),
            stalled_cycles=stalled, master_u=commitment(self.whole.d, self.last_master["x"]).u.tolist(),
            last_probe=self.last_probe, cut_feedback=cut_feedback,
            hourly_master_prediction=(self.last_master["hourly_recourse"].tolist()
                                      if "hourly_recourse" in self.last_master else None),
            hourly_cost_definition=("Scenario-weighted fuel, shedding and curtailment penalties"
                if getattr(self.whole.d, "model_profile", "") == "azema_sms"
                else "Scenario-weighted fuel plus the shared cold startup premium"),
            hourly_cut_feedback=[{key: row.get(key) for key in ("cut_id", "source", "hour", "master_violation")}
                                 for row in self.hourly.cuts[-24:]],
            recent_probes=[{key: value for key, value in row.items() if key != "raw_u"}
                           for row in self.whole.diagnostics["bound_probes"][-8:]],
            bound_history=[{key: row[key] for key in ("lower", "upper", "elapsed_seconds")}
                           for row in self.result.certification_history[-8:]])
        return bound_advisor.request_body(self.whole.d, self.whole.scenarios, self.result,
                                          self.o, context, iteration)

    # 执行一个限时整数认证周期：同步割、优化主问题、读取 ObjBound，再评估主问题候选。
    # 计时覆盖建模与探测；超过全局截止时间的结果不再入账。
    def step(self):
        if self.whole.remaining() <= 0:
            raise TimeoutError
        if self.converged():
            return
        before = self.result.lower_bound
        started = perf_counter()
        if self.g is None:
            self._initialize()
        self._sync()
        if self.whole.remaining() <= 0:
            raise TimeoutError
        window = (30., 90., 180.)[self.window_index] if self.o.rts24_improvements else self.o.bound_slice_seconds
        self.g.Params.TimeLimit = min(window, self.whole.remaining())
        self.g.Params.BestBdStop = (self.result.objective - self.o.tolerance*max(1., abs(self.result.objective))
                                  - self.b.constant + 1e-5)
        if self.o.rts24_improvements:
            self.g.Params.MIPFocus = 3 if self.window_index else 0
            self.callback_saved = -np.inf
            self.active_floor = before
            self.whole.emit('master_solve_start')
            self.g.optimize(self._callback)
            if self.callback_error is not None:
                raise self.callback_error
        else:
            self.whole.emit('master_solve_start')
            self.g.optimize()
        elapsed = perf_counter()-started
        self.whole.timings["certification"] += elapsed
        names = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 11: "INTERRUPTED", 15: "USER_OBJ_LIMIT"}
        status = names.get(self.g.Status, f"STATUS_{self.g.Status}")
        row = dict(iteration=len(self.result.certification_history)+1, status=status,
                   master_seconds=elapsed, cuts=self.synced, bound_used=False,
                   hourly_cuts=len(self.hourly.cuts),
                   lower_before=before, lower=before, upper=self.result.objective,
                   elapsed_seconds=perf_counter()-self.whole.started)
        self.result.certification_history.append(row)
        if self.o.rts24_improvements:
            row.update(window_seconds=window, scenario_cuts=self.scenario_synced, floor_bound=before,
                       node_count=float(self.g.NodeCount), mip_focus=int(self.g.Params.MIPFocus))
        if self.whole.remaining() <= 0:
            raise TimeoutError
        if status not in ("OPTIMAL", "TIME_LIMIT", "INTERRUPTED", "USER_OBJ_LIMIT"):
            raise RuntimeError(f"Integer Benders master failed: {status}")
        # ObjBound 证明整个整数主问题的下界；ObjVal 只是某个候选的预测值，不能替代它。
        bound = float(self.g.ObjBound) + self.b.constant - 1e-5
        self.update_bound(bound)
        row.update(master_bound=bound, bound_used=np.isfinite(bound), lower=self.result.lower_bound)
        if self.o.rts24_improvements:
            self._advance_window(status, self.result.lower_bound-before)
            self._native_bound(bound, 'optimize_return')
        self.whole.emit("certified_bound")
        if self.o.rts24_improvements:
            self._pool_probes(row)
        elif self.g.SolCount:
            point = np.asarray(self.variables.X)
            raw = point[:self.master_data.nx]
            if not np.isfinite(raw).all() or np.max(np.abs(raw-np.rint(raw))) > 1e-7:
                raise RuntimeError("Integer master returned a nonbinary probe")
            x = np.rint(raw)
            prediction = float(self.g.ObjVal) + self.b.constant
            self.last_master = dict(x=x, predicted_cost=prediction,
                                    hourly_recourse=np.asarray(self.hourly_variables.X))
            self.last_probe = self._probe(x, "certificate", dict(
                proposal_id=f"certificate:{row['iteration']}", predicted_cost=prediction))
            if self.last_probe.get("outcome") == "infeasible" and not self.last_probe["cut_added"]:
                raise RuntimeError("Infeasible master probe produced no new certified cut")
            row.update(probe_hash=commitment_hash(commitment(self.whole.d, x)), predicted_cost=prediction,
                       actual_cost=self.last_probe["actual_cost"], cut_added=self.last_probe["cut_added"])
        if self.whole.remaining() <= 0:
            raise TimeoutError
        self.update_bound(self.result.lower_bound)
        row.update(lower=self.result.lower_bound, upper=self.result.objective,
                   elapsed_seconds=perf_counter()-self.whole.started)
        if self.o.rts24_improvements:
            self.whole.advice.after_certificate(before)
        self.whole.emit("certification")
        if self.o.verbose:
            print(f"certificate={row['iteration']} status={status} LB={row['lower']:.6f} "
                  f"UB={row['upper']:.6f} cuts={len(self.cuts)} hourly_cuts={len(self.hourly.cuts)}", flush=True)

    # 改进模式中，连续两次时间片耗尽且 LB 增益不足时，将窗口从 30 秒增至 90/180 秒。
    def _advance_window(self, status, gain):
        epsilon = max(1e-5, 1e-5*abs(self.result.objective))
        self.window_stalls = self.window_stalls+1 if status == 'TIME_LIMIT' and gain < epsilon else 0
        if self.window_stalls >= 2:
            self.window_index = min(2, self.window_index+1)
            self.window_stalls = 0

    # 保存有限的原生主问题下界及其割数量、常数项和继承下界信息，供审计区分证明来源。
    def _native_bound(self, bound, origin):
        if not np.isfinite(bound) or abs(bound) >= gp.GRB.INFINITY:
            return
        self.whole.diagnostics['native_bounds'].append(dict(origin=origin, master_bound=bound,
            constant=self.b.constant, floor_bound=self.active_floor, cuts=self.synced,
            hourly_cuts=self.hourly_synced, scenario_cuts=self.scenario_synced,
            elapsed_seconds=perf_counter()-self.whole.started))

    # 在整数求解回调中周期性记录 MIP_OBJBND；全局预算耗尽或发现异常时终止求解。
    # 回调异常暂存到 callback_error，由 step 在 optimize 返回后统一抛出。
    def _callback(self, g, where):
        try:
            if self.whole.remaining() <= 0:
                g.terminate()
                return
            if where != gp.GRB.Callback.MIP or perf_counter()-self.callback_saved < 5:
                return
            lower = float(g.cbGet(gp.GRB.Callback.MIP_OBJBND))+self.b.constant-1e-5
            if np.isfinite(lower) and abs(lower) < gp.GRB.INFINITY:
                self.update_bound(lower)
                self._native_bound(lower, 'MIP_OBJBND')
                self.whole.emit('native_bound')
            self.callback_saved = perf_counter()
        except Exception as error:
            self.callback_error = error
            g.terminate()

    # 从 Gurobi 解池按预测费用选择至多三个未评估的精确二进制探测点。
    # 解池方案必须重新做全天连续调度，预测目标值不能直接替代原问题费用或认证下界。
    def _pool_probes(self, row):
        candidates = []
        for index in range(self.g.SolCount):
            self.g.Params.SolutionNumber = index
            point = np.asarray(self.variables.Xn)
            raw = point[:self.master_data.nx]
            if not np.isfinite(raw).all() or np.max(abs(raw-np.rint(raw))) > 1e-7:
                raise RuntimeError('Nonbinary master pool solution')
            candidates.append((float(self.g.PoolObjVal)+self.b.constant, np.rint(raw),
                               np.asarray(self.hourly_variables.Xn)))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            cost, x, hourly = candidates[0]
            self.last_master = dict(x=x, predicted_cost=cost, hourly_recourse=hourly)
        count, seen = 0, set()
        for cost, x, hourly in candidates:
            key = x.astype(np.uint8).tobytes()
            if key in seen or key in self.whole.cache:
                continue
            if count == 3 or self.converged() or self.whole.remaining() <= 0:
                break
            seen.add(key); count += 1
            self.last_probe = self._probe(x, 'certificate', dict(
                proposal_id=f"certificate:{row['iteration']}:{count}", predicted_cost=cost))
            if self.last_probe.get('outcome') == 'infeasible' and not self.last_probe['cut_added']:
                raise RuntimeError('Infeasible pool probe has no separating cut')
            row.update(probe_hash=commitment_hash(commitment(self.whole.d, x)), predicted_cost=cost,
                       actual_cost=self.last_probe['actual_cost'], cut_added=self.last_probe['cut_added'])
        row['pool_probes'] = count

    # 释放本认证器持有的 Gurobi 模型与环境，避免保留许可证会话和原生资源。
    def close(self):
        if self.g is not None:
            self.g.dispose()
        if self.env is not None:
            self.env.dispose()
