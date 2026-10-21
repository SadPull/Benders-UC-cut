"""Upper-bound advice bursts and measured candidate feedback."""
from copy import deepcopy
from dataclasses import replace
import json
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from . import advisor_request, llm

TIER = dict(regular="常规层", escalation="升级层")
TRANSPORT = dict(codex_exec="Codex 独立任务", codex_subagent="手动文件交换", http="HTTP")


def transport(o):
    return "codex_exec" if o.llm_codex_command else "codex_subagent" if o.llm_agent_directory else "http"


# 升级层随检查点保存的状态：是否武装及触发调用、连续失败、停用原因、调用次数和最近一次有效改进的时刻。
def escalation_state():
    return dict(armed=False, armed_by=None, consecutive_failures=0, disabled=None, calls=0,
                last_improved_seconds=None)


class UpperAdvisor:
    def __init__(self, whole):
        self.whole, self.cert, self.o, self.result = whole, whole.certifier, whole.o, whole.result
        # worker/pending/baseline 属于常规层；升级层的在途请求另放在 escalation，两层并行、互不阻塞。
        self.worker, self.pending, self.baseline = None, None, None
        self.escalation = NS(worker=None, pending=None, baseline=None)
        self.event_row = None
        # 会话身份、重置次数和停用原因都放在 state 中，随检查点保存。
        self.state = dict(version=2, phase="sa", stalled=dict(upper=0), sa_rounds=0,
                          batch_id=0, batch=None, consecutive_failures=0,
                          thread_id=None, session_resets=0, disabled=None, escalation=escalation_state())
        whole.diagnostics.update(advisor_state=self.state, advisor_rounds=[], llm_batches=[],
                                 upper_advice_feedback=None)

    @property
    def thread_id(self):
        return self.state["thread_id"]

    @thread_id.setter
    def thread_id(self, value):
        self.state["thread_id"] = value

    # 进度记录据 event_row 找到本次事件对应的调用；两层并行时它不一定是 llm_history 的最后一行。
    def emit(self, event, row):
        self.event_row = row
        self.whole.emit(event)

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
        # base_u 是提示中模型看到的 incumbent_u；增量候选只有配合它才能还原成完整方案，随回执落盘。
        self.pending = dict(call_id=f"llm_upper:{iteration}", stage="advice_batch", **routing,
            tier="regular", requested_model=self.o.llm_model, transport=transport(self.o),
            deadline_seconds=self.o.llm_timeout,
            status="RUNNING", outer=batch["outer"], round=batch["round"], accepted=0, proposals=[],
            base_u=llm.prompt_base(body), prompt_mode="delta" if continuation else "full",
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
            self.arm(self.pending)
        self.emit("llm_request", self.pending)
        if self.o.verbose:
            print(f"[LLM 请求] 常规层 方式={TRANSPORT[transport(self.o)]} 模型={self.o.llm_model} 推理=max "
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
        if self.consume(self.pending, proposals, info, self.baseline, searches, self.record_failure):
            self.arm(self.pending)
        return True

    # 两层共用的回复处理：落回执、累计后台时长、更新失败计数、评估候选并记下前后 UB。
    # 消费时总预算已用完则只记 TIME_LIMIT、不评估，返回 False；否则返回 True。
    def consume(self, row, proposals, info, baseline, searches, on_status):
        row.update(info, consumed_seconds=perf_counter()-self.whole.started)
        if self.o.verbose:
            print(f"[LLM 回复] {TIER[row.get('tier', 'regular')]} 调用={row.get('call_id')} 状态={info['status']} "
                  f"格式有效候选={len(proposals)} 耗时={info['background_seconds']:.2f}秒 "
                  f"日志={info.get('request_directory', '见结果回执')}", flush=True)
        self.whole.diagnostics["llm_background_seconds"] += info["background_seconds"]
        if self.whole.remaining() <= 0:
            row.update(status="TIME_LIMIT", accepted=0)
            return False
        on_status()
        before_upper = self.result.objective
        if info["status"] == "OK":
            self.whole.consume_upper(proposals, row, searches, baseline)
        row.update(upper_before_probes=before_upper, upper_after_probes=self.result.objective,
            direct_incumbent_decrease=sum(r.get("incumbent_decrease", 0.) for r in info.get("proposals", [])),
            numerical_evaluation_completed=any(r.get("continuous_dispatch_completed")
                                              for r in info.get("proposals", [])))
        self.emit("llm_response", row)
        return True

    # 会话失效或连续两次失败时，先丢弃会话身份新开一次；再失败就停用 LLM，主算法继续 SA。
    def record_failure(self):
        failed = self.pending['status'] != 'OK'
        self.state['consecutive_failures'] = self.state['consecutive_failures']+1 if failed else 0
        self.emit('llm_call_status', self.pending)
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

    # 常规层一次调用降幅低于 llm_escalation_gain 或调用失败时武装升级层。取消的调用、超期消费的回复，
    # 以及早于最近一次升级层有效改进发出的请求不武装：先让常规层在新方案上试一次。
    def arm(self, row):
        s = self.state['escalation']
        if not self.escalation_enabled() or s['disabled'] or row['status'] == 'CANCELLED':
            return
        if s['last_improved_seconds'] is not None and row['started_seconds'] < s['last_improved_seconds']:
            return
        if row['status'] == 'OK':
            before = row['upper_before_probes']
            gain = (before-row['upper_after_probes'])/max(1., abs(before))
            if gain >= self.o.llm_escalation_gain:
                return
            reason = f"降幅 {100*gain:.3f}% 低于 {100*self.o.llm_escalation_gain:g}%"
        else:
            reason = f"状态 {row['status']}"
        s.update(armed=True, armed_by=row['call_id'])
        if self.o.verbose:
            print(f"[LLM 升级] 常规层 {row['call_id']} {reason}；升级层待发起。", flush=True)

    def advance(self, searches):
        """Each tier consumes at most one reply and launches at most one request; never waits."""
        self.advance_batch(searches)
        self.advance_escalation(searches)

    def advance_batch(self, searches):
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
        self._end_batch()
        batch.update(status=reason, upper_after=self.result.objective, lower_after=self.result.lower_bound,
                     sa_calls_at_end=self.whole.diagnostics["sa_calls"],
                     ended_seconds=perf_counter()-self.whole.started)
        self.state.update(phase="sa", batch=None, stalled=dict(upper=0))

    def escalation_enabled(self):
        return self.o.llm_enabled and bool(self.o.llm_escalation_model)

    # 升级层本次可用秒数：剩余预算扣除预留后不足 min_seconds 时为 None，否则取其与单次上限的较小值。
    def escalation_window(self):
        window = self.whole.remaining()-self.o.llm_reserve_seconds
        if window <= 0 or window < self.o.llm_escalation_min_seconds:
            return None
        return min(self.o.llm_escalation_timeout, window)

    def escalation_ready(self):
        s = self.state['escalation']
        return (s['armed'] and not s['disabled'] and self.escalation.worker is None
                and not self.cert.converged() and self.escalation_window() is not None
                and (self.o.llm_max_calls is None or self.result.llm_calls < self.o.llm_max_calls))

    # 每步先收升级层回复，再按武装状态发起新请求；与常规批次无关，收敛或超时后的在途请求由 close 取消。
    def advance_escalation(self, searches):
        if not self.escalation_enabled() or self.cert.converged() or self.whole.remaining() <= 0:
            return
        if self.escalation.worker is not None:
            self.poll_escalation(searches)
        if self.escalation_ready():
            self.start_escalation()

    # 升级层发送完整提示、不续接会话，在隔离的临时 CODEX_HOME 中运行；调用前快照单独保存，
    # 回复到达时若 incumbent 已变化，由 consume_upper 的移植步骤把改动机组移到当前方案上。
    def start_escalation(self):
        s, lane = self.state['escalation'], self.escalation
        iteration = self.result.llm_calls+1
        context, bases, baseline = self.whole.upper_context()
        context['continuation'] = dict(session_id=None, completed_calls=self.result.llm_calls,
                                       latest_snapshot_authoritative=True)
        o = replace(self.o, llm_model=self.o.llm_escalation_model, llm_codex_command=self.o.llm_escalation_command,
                    llm_agent_directory=self.o.llm_escalation_directory)
        body = llm.request_body(self.whole.d, self.whole.scenarios, self.result, o, context, bases, iteration)
        seconds = self.escalation_window()
        if seconds is None:
            return
        o.llm_timeout = seconds
        lane.baseline = deepcopy(baseline)
        triggered_by = s['armed_by']
        s.update(armed=False, armed_by=None, calls=s['calls']+1)
        self.result.llm_calls = iteration
        routing = dict(channel="upper", tier="escalation",
                       phase="resume" if self.whole.diagnostics.get('resume') else "formal")
        prompt = json.loads(body["input"])
        prompt["advice_routing"] = routing
        body["input"] = json.dumps(prompt, separators=(",", ":"), allow_nan=False)
        lane.pending = row = dict(call_id=f"llm_escalation:{iteration}", stage="escalation", **routing,
            triggered_by=triggered_by, requested_model=o.llm_model, transport=transport(o),
            deadline_seconds=seconds, status="RUNNING", accepted=0, proposals=[],
            base_u=llm.prompt_base(body), prompt_mode="full",
            request_upper=self.result.objective, request_lower=self.result.lower_bound,
            started_seconds=perf_counter()-self.whole.started)
        self.result.llm_history.append(row)
        if self.whole.diagnostics["llm_calls"] is not self.result.llm_history:
            self.whole.diagnostics["llm_calls"].append(row)
        try:
            lane.worker = advisor_request.AdvisorRequest(self.whole.d, o, body, iteration, seconds,
                                                         session_id=None, isolated=True)
        except Exception as error:
            row.update(status="FAILED", message=f"{type(error).__name__}: {error}")
            self.escalation_failure()
        self.emit("llm_request", row)
        if self.o.verbose:
            print(f"[LLM 请求] 升级层 方式={TRANSPORT[transport(o)]}（隔离配置、不续接） 模型={o.llm_model} "
                  f"推理=max 调用次数={iteration} 触发={triggered_by} 时限={seconds:.0f}秒 "
                  f"状态={row['status']}", flush=True)

    def poll_escalation(self, searches):
        lane, s = self.escalation, self.state['escalation']
        reply = lane.worker.poll()
        if reply is None:
            return
        lane.worker = None
        proposals, info = reply
        if not self.consume(lane.pending, proposals, info, lane.baseline, searches, self.escalation_failure):
            return
        # 有效改进后清除期间累积的武装，让常规层先在新方案上试；没有改进时保留，可立即再发。
        lane.pending['improved'] = self.improved(lane.pending['upper_before_probes'])
        if lane.pending['improved']:
            s.update(armed=False, armed_by=None, last_improved_seconds=perf_counter()-self.whole.started)

    # 升级层不续接会话：连续两次失败只停用升级层，常规层照常。
    def escalation_failure(self):
        row, s = self.escalation.pending, self.state['escalation']
        s['consecutive_failures'] = s['consecutive_failures']+1 if row['status'] != 'OK' else 0
        self.emit('llm_call_status', row)
        if s['consecutive_failures'] >= 2 and not s['disabled']:
            s.update(disabled='Two consecutive escalation calls failed', armed=False, armed_by=None)
            self.whole.diagnostics['llm_escalation_disabled'] = dict(
                reason=s['disabled'], seconds=perf_counter()-self.whole.started)
            if self.o.verbose:
                print(f"[LLM 停用] 升级层：{s['disabled']}；常规层照常。", flush=True)

    def restore(self, diagnostic):
        saved = diagnostic["advisor_state"]
        if saved["version"] != 2:
            raise ValueError("Only the current advice scheduling state is supported")
        self.state.clear()
        self.state.update(deepcopy(saved))
        self.state.setdefault('thread_id', None)
        self.state.setdefault('session_resets', 0)
        self.state.setdefault('disabled', None)
        self.state['escalation'] = dict(escalation_state(), **self.state.get('escalation', {}))
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
        # 没有会话身份且未主动重置过时，取常规层最后一次成功会话；升级层的临时会话不可续接。
        if self.thread_id is None and not self.state['session_resets']:
            self.thread_id = next((row['thread_id'] for row in reversed(self.result.llm_history)
                                   if row['status'] == 'OK' and row.get('thread_id')
                                   and row.get('tier', 'regular') == 'regular'), None)

    def _cancel(self, worker, row):
        elapsed = perf_counter()-worker.started
        self.whole.diagnostics["llm_background_seconds"] += elapsed
        worker.close()
        row.update(status="CANCELLED", message="SearchFinished", accepted=0,
            seconds=elapsed, background_seconds=elapsed, **worker.receipt)

    # 常规批次收尾：只取消常规层在途请求，升级层不受影响。
    def _end_batch(self):
        if self.worker is not None:
            self._cancel(self.worker, self.pending)
            self.worker = None
        batch = self.state['batch']
        if batch is not None and batch['status'] == 'RUNNING':
            batch.update(status='CANCELLED', upper_after=self.result.objective,
                lower_after=self.result.lower_bound, sa_calls_at_end=self.whole.diagnostics['sa_calls'],
                ended_seconds=perf_counter()-self.whole.started)
        self.whole.diagnostics["upper_advice_feedback"] = self.whole.feedback

    # 求解结束时调用一次：收尾常规批次，并取消升级层在途请求。
    def close(self):
        self._end_batch()
        if self.escalation.worker is not None:
            self._cancel(self.escalation.worker, self.escalation.pending)
            self.escalation.worker = None


# 关联候选回执、调用前方案与已完成评估，返回费用变化、修复距离和缓存命中反馈。
# 没有实际调度结果时不编造费用；历史或重复方案不记为本次刷新 incumbent。
# shortfall 给定时，对调度不可行的候选附上可达出力低于负荷的小时及缺口（MW），供模型修正；
# 按模型原样提出的 raw_u 计算，没有时退回实际评估的修复后方案。
def summarize(advice, baseline_u, baseline, evaluations, previous_ids, shortfall=None):
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
                   reachable_shortfall=(shortfall(p['raw_u'] if p.get('raw_u') is not None else r['u'])
                                        if shortfall and r and not r['feasible']
                                        and (p.get('raw_u') is not None or r.get('u') is not None) else None),
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
