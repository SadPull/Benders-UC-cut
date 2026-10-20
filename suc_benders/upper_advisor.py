"""Upper-bound advice bursts and measured candidate feedback."""
from copy import deepcopy
import json
from time import perf_counter

import numpy as np

from . import advisor_request, llm


class UpperAdvisor:
    def __init__(self, whole):
        self.whole, self.cert, self.o, self.result = whole, whole.certifier, whole.o, whole.result
        self.worker, self.pending, self.baseline = None, None, None
        # 会话身份、重置次数和停用原因都放在 state 中，随检查点保存。
        self.state = dict(version=2, phase="sa", stalled=dict(upper=0), sa_rounds=0,
                          batch_id=0, batch=None, consecutive_failures=0,
                          thread_id=None, session_resets=0, disabled=None)
        whole.diagnostics.update(advisor_state=self.state, advisor_rounds=[], llm_batches=[],
                                 upper_advice_feedback=None)

    @property
    def thread_id(self):
        return self.state["thread_id"]

    @thread_id.setter
    def thread_id(self, value):
        self.state["thread_id"] = value

    def available(self):
        return (self.o.llm_enabled and not self.state["disabled"]
                and self.whole.remaining() >= self.o.llm_timeout+self.o.llm_reserve_seconds
                and (self.o.llm_max_calls is None or self.result.llm_calls < self.o.llm_max_calls))

    # 冷启动初始化后立即开一批；请求由后续 advance 异步发出，不阻塞 SA。
    def kickoff(self):
        if (self.o.llm_first_call_at_start and self.result.llm_calls == 0 and self.state["batch"] is None
                and self.available() and not self.cert.converged()):
            self.begin(trigger="start")

    def improved(self, upper):
        return upper-self.result.objective >= .0005*max(1., abs(upper))

    def after_round(self, upper, sa_before, searches, outer, inner, allow=True):
        if self.whole.diagnostics["sa_calls"] <= sa_before:
            return
        self.state["sa_rounds"] += 1
        self.state["stalled"]["upper"] = (0 if self.improved(upper) else self.state["stalled"]["upper"]+1)
        self.whole.diagnostics["advisor_rounds"].append(dict(
            outer=outer, round=inner, sa_round=self.state["sa_rounds"], upper_before=upper,
            upper=self.result.objective, lower=self.result.lower_bound,
            stalled=self.state["stalled"].copy()))
        if (allow and self.state['batch'] is None and self.available() and not self.cert.converged()
                and self.state["stalled"]["upper"] >= self.o.llm_patience):
            self.begin(outer=outer, inner=inner)
        self.advance(searches)

    def begin(self, *, outer=None, inner=None, trigger="stall"):
        if self.state["phase"] != "sa" or self.worker is not None:
            raise RuntimeError("An advice batch is already active")
        self.state["batch_id"] += 1
        batch = dict(batch_id=self.state["batch_id"], channels=["upper"], trigger=trigger,
            counts=dict(upper=0), done=dict(upper=False),
            upper_before=self.result.objective, lower_before=self.result.lower_bound,
            trigger_stalled=self.state["stalled"].copy(), trigger_sa_round=self.state["sa_rounds"],
            sa_calls_at_start=self.whole.diagnostics["sa_calls"], outer=outer, round=inner,
            status="RUNNING", started_seconds=perf_counter()-self.whole.started)
        self.state.update(phase="llm", batch=batch)
        self.whole.diagnostics["llm_batches"].append(batch)
        if self.o.verbose:
            reason = ("初始化完成" if trigger == "start" else
                      f"连续 {batch['trigger_stalled']['upper']} 个完整 SA 轮次改善不足")
            print(f"[LLM 触发] {reason}，开始第 {batch['batch_id']} 批上界建议。", flush=True)

    def start_request(self):
        batch = self.state["batch"]
        if self.worker is not None or not self.available() or batch["counts"]["upper"] >= 3:
            return
        iteration = self.result.llm_calls+1
        context, bases, self.baseline = self.whole.upper_context()
        self.baseline = deepcopy(self.baseline)
        context['continuation'] = dict(session_id=self.thread_id, completed_calls=self.result.llm_calls,
                                       latest_snapshot_authoritative=True)
        continuation = self.thread_id is not None
        body = llm.request_body(self.whole.d, self.whole.scenarios, self.result,
                                self.o, context, bases, iteration, continuation)
        if not self.available():
            return
        batch["counts"]["upper"] += 1
        self.result.llm_calls = iteration
        routing = dict(channel="upper", batch_id=batch["batch_id"],
                       batch_call=batch["counts"]["upper"],
                       phase="resume" if self.whole.diagnostics.get('resume') else "formal")
        prompt = json.loads(body["input"])
        prompt["advice_routing"] = routing
        body["input"] = json.dumps(prompt, separators=(",", ":"), allow_nan=False)
        self.pending = dict(call_id=f"llm_upper:{iteration}", stage="advice_batch", **routing,
            status="RUNNING", outer=batch["outer"], round=batch["round"], accepted=0, proposals=[],
            prompt_mode="delta" if continuation else "full",
            trigger_stalled=batch["trigger_stalled"].copy(),
            request_upper=self.result.objective, request_lower=self.result.lower_bound,
            started_seconds=perf_counter()-self.whole.started)
        self.result.llm_history.append(self.pending)
        if self.whole.diagnostics["llm_calls"] is not self.result.llm_history:
            self.whole.diagnostics["llm_calls"].append(self.pending)
        try:
            self.worker = advisor_request.AdvisorRequest(self.whole.d, self.o, body, iteration,
                self.o.llm_timeout, session_id=self.thread_id)
        except Exception as error:
            self.pending.update(status="FAILED", message=type(error).__name__)
            self.record_failure()
        self.whole.emit("llm_request")
        if self.o.verbose:
            transport = ('Codex 独立任务' if self.o.llm_codex_command else
                         '手动文件交换' if self.o.llm_agent_directory else 'HTTP')
            print(f"[LLM 请求] 方式={transport} 模型={self.o.llm_model} 推理=max "
                  f"批次={batch['batch_id']} 调用次数={iteration} "
                  f"本批次数={batch['counts']['upper']} 状态={self.pending['status']}", flush=True)

    def poll(self, searches):
        if self.worker is None:
            return False
        reply = self.worker.poll()
        if reply is None:
            return False
        self.worker = None
        proposals, info = reply
        if self.o.llm_codex_command and info['status'] == 'OK':
            if not info.get('thread_id') or (self.thread_id is not None and info['thread_id'] != self.thread_id):
                proposals = []
                info.update(status='SESSION_FAILED', message='Codex session identity mismatch', accepted=0)
            else:
                self.thread_id = info['thread_id']
        self.pending.update(info, consumed_seconds=perf_counter()-self.whole.started)
        if self.o.verbose:
            print(f"[LLM 回复] 调用次数={self.result.llm_calls} 状态={info['status']} "
                  f"格式有效候选={len(proposals)} 耗时={info['background_seconds']:.2f}秒 "
                  f"日志={info.get('request_directory', '见结果回执')}", flush=True)
        self.whole.diagnostics["llm_background_seconds"] += info["background_seconds"]
        if self.whole.remaining() <= 0:
            self.pending.update(status="TIME_LIMIT", accepted=0)
            return True
        self.record_failure()
        before_upper = self.result.objective
        if info["status"] == "OK":
            self.whole.consume_upper(proposals, self.pending, searches, self.baseline)
        self.pending.update(upper_before_probes=before_upper, upper_after_probes=self.result.objective,
            direct_incumbent_decrease=sum(r.get("incumbent_decrease", 0.) for r in info.get("proposals", [])),
            numerical_evaluation_completed=any(r.get("continuous_dispatch_completed")
                                              for r in info.get("proposals", [])))
        self.whole.emit("llm_response")
        return True

    # 会话失效或连续两次失败时，先丢弃会话身份新开一次；再失败就停用 LLM，主算法继续 SA。
    def record_failure(self):
        failed = self.pending['status'] != 'OK'
        self.state['consecutive_failures'] = self.state['consecutive_failures']+1 if failed else 0
        self.whole.emit('llm_call_status')
        if self.pending['status'] == 'SESSION_FAILED':
            reason = self.pending.get('message') or 'Codex session unavailable'
        elif self.state['consecutive_failures'] >= 2:
            reason = 'Two consecutive upper calls failed'
        else:
            return
        if self.state['session_resets'] < 1:
            self.state.update(session_resets=self.state['session_resets']+1, consecutive_failures=0)
            self.thread_id = None
            if self.o.verbose:
                print(f"[LLM 会话] {reason}；下一次请求新开会话。", flush=True)
            return
        self.state['disabled'] = reason
        self.whole.diagnostics['llm_disabled'] = dict(reason=reason, seconds=perf_counter()-self.whole.started)
        if self.o.verbose:
            print(f"[LLM 停用] {reason}；新会话后仍失败，剩余预算只用数值搜索。", flush=True)

    def advance(self, searches):
        """Consume at most one reply and launch at most one request; never wait."""
        batch = self.state["batch"]
        if batch is None:
            return
        batch['status'] = 'RUNNING'
        if self.cert.converged():
            reason = 'CONVERGED'
        elif self.whole.remaining() <= 0:
            reason = "TIME_LIMIT"
        else:
            if self.worker is not None and not self.poll(searches):
                return
            if self.whole.remaining() <= 0:
                reason = 'TIME_LIMIT'
            elif not self.available():
                reason = ('LLM_DISABLED' if self.state['disabled'] else
                          'CALL_LIMIT' if self.o.llm_max_calls is not None
                          and self.result.llm_calls >= self.o.llm_max_calls else 'LLM_WINDOW_TOO_SHORT')
            elif self.improved(batch['upper_before']):
                batch['done']['upper'] = True
                reason = 'RETURN_TO_SA'
            elif batch['counts']['upper'] >= 3:
                reason = 'BATCH_LIMIT'
            else:
                self.start_request()
                return
        self.close()
        batch.update(status=reason, upper_after=self.result.objective, lower_after=self.result.lower_bound,
                     sa_calls_at_end=self.whole.diagnostics["sa_calls"],
                     ended_seconds=perf_counter()-self.whole.started)
        self.state.update(phase="sa", batch=None, stalled=dict(upper=0))

    def restore(self, diagnostic):
        saved = diagnostic["advisor_state"]
        if saved["version"] != 2:
            raise ValueError("Only the current advice scheduling state is supported")
        self.state.clear()
        self.state.update(deepcopy(saved))
        self.state.setdefault('thread_id', None)
        self.state.setdefault('session_resets', 0)
        self.state.setdefault('disabled', None)
        for name in ("advisor_rounds", "llm_batches"):
            self.whole.diagnostics[name] = deepcopy(diagnostic[name])
        if self.state["batch"] is not None:
            batch = next(b for b in self.whole.diagnostics["llm_batches"]
                         if b["batch_id"] == self.state["batch_id"])
            if batch['channels'] != ['upper'] or batch.get('validation', False):
                raise ValueError('Recovered advice must be an upper search batch')
            self.state["batch"] = batch
        for row in self.result.llm_history:
            if row["status"] == "RUNNING":
                row.update(status="CANCELLED", message="ResumeWorkerUnavailable", accepted=0)
        self.whole.feedback = deepcopy(diagnostic["upper_advice_feedback"])
        # 没有会话身份且未主动重置过时，取最后一次成功会话；找不到则下一次请求新开会话。
        if self.thread_id is None and not self.state['session_resets']:
            self.thread_id = next((row['thread_id'] for row in reversed(self.result.llm_history)
                                   if row['status'] == 'OK' and row.get('thread_id')), None)

    def close(self):
        if self.worker is not None:
            elapsed = perf_counter()-self.worker.started
            self.whole.diagnostics["llm_background_seconds"] += elapsed
            self.pending.update(status="CANCELLED", message="SearchFinished", accepted=0,
                seconds=elapsed, background_seconds=elapsed, **self.worker.receipt)
            self.worker.close()
            self.worker = None
        batch = self.state['batch']
        if batch is not None and batch['status'] == 'RUNNING':
            batch.update(status='CANCELLED', upper_after=self.result.objective,
                lower_after=self.result.lower_bound, sa_calls_at_end=self.whole.diagnostics['sa_calls'],
                ended_seconds=perf_counter()-self.whole.started)
        self.whole.diagnostics["upper_advice_feedback"] = self.whole.feedback


# 关联候选回执、调用前方案与已完成评估，返回费用变化、修复距离和缓存命中反馈。
# 没有实际调度结果时不编造费用；历史或重复方案不记为本次刷新 incumbent。
def summarize(advice, baseline_u, baseline, evaluations, previous_ids):
    measured = {r["commitment_hash"]: r for r in evaluations}
    rows = []
    for p in advice.diagnostics.get("proposals", []):
        identity = p.get("commitment_id")
        r = measured.get(identity)
        cached = identity in previous_ids or p.get("outcome") == "duplicate"
        row = dict(proposal_id=p["proposal_id"], commitment_id=identity,
                   proposed_changes=changes(baseline_u, p.get("raw_u")),
                   repair_changes=changes(p.get("raw_u"), p.get("repaired_u")),
                   repair_bits=p.get("hamming_u"), outcome=p.get("outcome"),
                   invalid_reason=p.get("invalid_reason"),
                   duplicate=cached, cache_hit=bool(cached and r),
                   continuous_dispatch_completed=bool(r or p.get("local_dispatch_completed")),
                   local_feasible=p.get("local_feasible"),
                   validation_completed=r is not None, feasible=r["feasible"] if r else None,
                   repair_reason=(dict(known_raw_violations=p.get('raw_constraint_violations', []),
                       note='Known violations of the original proposal; not a full explanation of every repair edit.')
                       if p.get('hamming_u') or p.get('raw_constraint_violations') else None),
                   infeasibility_reason=(r['dispatch_status'] if r and not r['feasible'] else None),
                   baseline_cost=baseline, costs=None, cost_change=None,
                   refreshed_incumbent=bool(r and r["improved"] and not cached))
        if r and r["feasible"]:
            costs = dict(total=r["objective"], fixed=r["fixed_cost"], expected_fuel=r["expected_fuel_cost"])
            row["costs"] = costs
            if baseline is not None:
                row["cost_change"] = {k: costs[k]-baseline[k] for k in costs}
        rows.append(row)
    return dict(call_id=advice.diagnostics.get("call_id"), failed=advice.failed,
                message=advice.message, proposals=rows)


# 把两张机组 × 小时矩阵的差异合并成连续区间；缺失、形状不符或非有限值不计入变化。
# 输出的机组和小时编号从 1 开始，end_hour 表示包含在区间内的最后一小时。
def changes(before, after):
    if before is None or after is None:
        return []
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape:
        return []
    rows = []
    for i in range(len(before)):
        for t in np.flatnonzero(before[i] != after[i]):
            old, new = float(before[i, t]), float(after[i, t])
            if not np.isfinite(old) or not np.isfinite(new):
                continue
            if rows and rows[-1]["unit"] == i+1 and rows[-1]["end_hour"] == t and rows[-1]["from"] == old and rows[-1]["to"] == new:
                rows[-1]["end_hour"] = int(t+1)
            else:
                rows.append(dict(unit=i+1, start_hour=int(t+1), end_hour=int(t+1), **{"from": old, "to": new}))
    return rows
