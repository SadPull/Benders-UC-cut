"""Separate upper/lower advice bursts, triggered only by completed SA rounds."""
# 模块职责：分别跟踪上界降本与下界探测的停滞，管理两通道批次和唯一在途请求。
# 常规模式按实际采样的完整 SA 轮触发；改进模式独立统计下界认证周期并非阻塞调度。
from copy import deepcopy
import json
from time import perf_counter, sleep

from . import bound_advisor, llm


# 上下界 LLM 调度状态机；建议由数值求解器评估，模型不能直接修改认证费用或界。
class DualAdvisor:
    # 初始化两侧停滞计数、批次记录和唯一工作进程；改进模式使用 version=2 的独立触发状态。
    def __init__(self, whole):
        self.whole, self.cert, self.o, self.result = whole, whole.certifier, whole.o, whole.result
        self.worker, self.pending, self.baseline = None, None, None
        self.state = dict(version=1, phase="sa", stalled=dict(upper=0, lower=0),
                          sa_rounds=0, batch_id=0, batch=None)
        whole.diagnostics.update(advisor_state=self.state, advisor_rounds=[], llm_batches=[])
        if self.o.rts24_improvements:
            self.state.update(version=2, certification_cycles=0, last_served='upper')

    # 同时检查 LLM 开关、全局剩余时间和两侧共享的请求次数上限。
    def available(self):
        return (self.o.llm_enabled and self.whole.remaining() > 0
                and (self.o.llm_max_calls is None or self.result.llm_calls < self.o.llm_max_calls))

    # 根据所选模式的阈值判断本通道是否显著改善：上界比较降本，下界比较 LB 增长。
    def improved(self, channel, upper, lower):
        if self.o.rts24_improvements:
            gain = upper-self.result.objective if channel == 'upper' else self.result.lower_bound-lower
            return gain >= max(1e-5, 1e-5*abs(self.result.objective))
        if channel == "upper":
            return upper-self.result.objective >= .0005*max(1., abs(upper))
        return self.result.lower_bound-lower >= 1e-5+self.o.tolerance*max(1., abs(self.result.objective))

    # 仅在完整 SA 轮实际新增采样后更新停滞统计，并按模式触发批次或执行调度 tick。
    # 中断轮与没有新采样的轮不通过此方法增加停滞计数。
    def after_round(self, upper, lower, sa_before, searches, outer, inner, allow=True):
        if self.whole.diagnostics["sa_calls"] <= sa_before:
            return
        self.state["sa_rounds"] += 1
        for channel in (('upper',) if self.o.rts24_improvements else ("upper", "lower")):
            self.state["stalled"][channel] = (0 if self.improved(channel, upper, lower)
                                             else self.state["stalled"][channel]+1)
        self.whole.diagnostics["advisor_rounds"].append(dict(
            outer=outer, round=inner, sa_round=self.state["sa_rounds"], upper_before=upper,
            lower_before=lower, upper=self.result.objective, lower=self.result.lower_bound,
            stalled=self.state["stalled"].copy()))
        if not allow or not self.available() or self.cert.converged():
            return
        if self.o.rts24_improvements:
            self.service(searches)
            return
        channels = [c for c in ("upper", "lower") if self.state["stalled"][c] >= self.o.llm_patience
                    and (c == "upper" or self.cert.last_master is not None)]
        if channels:
            self.begin(channels, outer=outer, inner=inner)
            self.run_batch(searches)

    # 改进模式专用：在认证周期后按实际 LB 增长更新下界停滞次数。
    def after_certificate(self, lower_before):
        self.state['certification_cycles'] += 1
        self.state['stalled']['lower'] = (0 if self.improved('lower', self.result.objective, lower_before)
                                          else self.state['stalled']['lower']+1)

    # 改进模式的非阻塞调度 tick：先消费在途回复，再按停滞条件公平选择上下界通道。
    # 只在数值求解之间调用，保证评估与模型修改仍在父线程顺序执行。
    def service(self, searches):
        """One nonblocking scheduling tick, called only outside numeric solves."""
        if not self.o.rts24_improvements or self.cert.converged() or self.whole.remaining() <= 0:
            return
        if self.worker is not None:
            if not self.poll(searches):
                return
            channel = self.pending['channel']
            self.state['last_served'] = channel
            if self.improved(channel, self.pending['request_upper'], self.pending['request_lower']):
                self.state['stalled'][channel] = 0
            if self.state['batch']['counts'][channel] >= 3:
                self.state['stalled'][channel] = 0
                self.state['batch']['done'][channel] = True
        if not self.available() or self.cert.converged():
            return
        batch = self.state['batch']
        due = [c for c in ('lower', 'upper') if self.state['stalled'][c] >= self.o.llm_patience
               and (c == 'upper' or self.cert.last_master is not None)
               and (batch is None or not batch['done'][c])]
        if not due:
            if batch is not None:
                batch.update(status='RETURN_TO_SEARCH', upper_after=self.result.objective,
                    lower_after=self.result.lower_bound, ended_seconds=perf_counter()-self.whole.started,
                    sa_calls_at_end=self.whole.diagnostics['sa_calls'])
                self.state.update(phase='sa', batch=None)
            return
        if batch is None:
            self.begin(['upper', 'lower'])
        order = ('lower', 'upper') if self.state['last_served'] == 'upper' else ('upper', 'lower')
        self.start_request(next(c for c in order if c in due))

    # 创建带触发原因和起始上下界快照的新批次，formal 与 validation 阶段分别标记。
    def begin(self, channels, *, outer=None, inner=None, validation=False):
        if self.state["phase"] != "sa" or self.worker is not None:
            raise RuntimeError("An advice batch is already active")
        self.state["batch_id"] += 1
        batch = dict(batch_id=self.state["batch_id"], channels=list(channels),
            counts={c: 0 for c in channels}, done={c: False for c in channels}, last_channel="lower",
            upper_before=self.result.objective, lower_before=self.result.lower_bound,
            trigger_stalled=self.state["stalled"].copy(), trigger_sa_round=self.state["sa_rounds"],
            sa_calls_at_start=self.whole.diagnostics["sa_calls"],
            outer=outer, round=inner, validation=validation, status="RUNNING",
            started_seconds=perf_counter()-self.whole.started)
        self.state.update(phase="llm", batch=batch)
        self.whole.diagnostics["llm_batches"].append(batch)

    # 为所选通道刷新上下文、登记路由和回执，再发起一个限时后台请求。
    # 发出请求前即消耗全局与批次额度，因此传输失败也占一次；每侧每批最多三次。
    def start_request(self, channel):
        batch = self.state["batch"]
        if self.worker is not None or not self.available() or batch["counts"][channel] >= 3:
            return
        iteration = self.result.llm_calls+1
        if channel == "upper":
            context, bases, self.baseline = self.whole.upper_context()
            body = llm.request_body(self.whole.d, self.whole.scenarios, self.result,
                                    self.o, context, bases, iteration)
        else:
            body = self.cert.advice_body(iteration, self.state["stalled"]["lower"])
        if self.whole.remaining() <= 0:
            return
        batch["counts"][channel] += 1
        batch["last_channel"] = channel
        self.result.llm_calls = iteration
        routing = dict(channel=channel, batch_id=batch["batch_id"],
                       batch_call=batch["counts"][channel], phase="validation" if batch["validation"] else "formal")
        prompt = json.loads(body["input"])
        prompt["advice_routing"] = routing
        body["input"] = json.dumps(prompt, separators=(",", ":"), allow_nan=False)
        self.pending = dict(call_id=f"llm_{channel}:{iteration}", stage="advice_batch", **routing,
            status="RUNNING", outer=batch["outer"], round=batch["round"], accepted=0, proposals=[],
            trigger_stalled=batch["trigger_stalled"].copy(),
            request_upper=self.result.objective, request_lower=self.result.lower_bound,
            started_seconds=perf_counter()-self.whole.started)
        self.result.llm_history.append(self.pending)
        if self.whole.diagnostics["llm_calls"] is not self.result.llm_history:
            self.whole.diagnostics["llm_calls"].append(self.pending)
        try:
            self.worker = bound_advisor.BoundAdvisor(self.whole.d, self.o, body, iteration,
                min(600., self.o.llm_timeout, self.whole.remaining()))
        except Exception as error:
            self.pending.update(status="FAILED", message=type(error).__name__)
            if self.o.rts24_improvements:
                self.state['last_served'] = channel
                if batch['counts'][channel] >= 3:
                    batch['done'][channel] = True
                    self.state['stalled'][channel] = 0
        self.whole.emit("llm_request")
        if self.o.verbose:
            print(f"advice channel={channel} batch={batch['batch_id']} call={iteration} "
                  f"batch_call={batch['counts'][channel]} status={self.pending['status']}", flush=True)

    # 非阻塞接收回复并交给对应数值评估路径，记录实际降本、割增量和评估是否完成。
    # 下界通道接受数按新增全天割计，上界通道接受数按通过全天验证的建议计，二者不能混用。
    def poll(self, searches):
        if self.worker is None:
            return False
        reply = self.worker.poll()
        if reply is None:
            return False
        self.worker = None
        proposals, info = reply
        if self.o.rts24_improvements and len(info.get('proposals', [])) > 2:
            proposals = []
            info.update(status='FAILED', message='TooManyCandidates', accepted=0)
        self.pending.update(info, consumed_seconds=perf_counter()-self.whole.started)
        self.whole.diagnostics["llm_background_seconds"] += info.get("background_seconds", info.get("seconds", 0.))
        if self.whole.remaining() <= 0:
            self.pending.update(status="TIME_LIMIT", accepted=0)
            return True
        channel = self.pending["channel"]
        before_upper, before_lower = self.result.objective, self.result.lower_bound
        cuts_before = len(self.cert.cuts)
        if info["status"] == "OK":
            if channel == "upper":
                self.whole.consume_upper(proposals, self.pending, searches, self.baseline)
            else:
                for raw, record in proposals:
                    if self.whole.remaining() <= 0:
                        record["outcome"] = "deadline"
                        continue
                    self.cert.probe_advice(raw, record)
                self.pending["accepted"] = sum(bool(r.get("cut_added")) for r in info.get("proposals", []))
        self.pending.update(upper_before_probes=before_upper, upper_after_probes=self.result.objective,
            lower_before_probes=before_lower, lower_after_probes=self.result.lower_bound,
            new_global_cuts=len(self.cert.cuts)-cuts_before,
            direct_incumbent_decrease=sum(r.get("incumbent_decrease", 0.) for r in info.get("proposals", [])))
        completed = any(r.get("continuous_dispatch_completed") for r in info.get("proposals", []))
        self.pending["numerical_evaluation_completed"] = completed
        self.whole.emit("llm_response")
        if self.state["batch"]["validation"] and completed:
            self.state["batch"]["done"][channel] = True
        if self.o.verbose:
            print(f"advice channel={channel} status={info['status']} "
                  f"new_cuts={self.pending['new_global_cuts']} UB={self.result.objective:.6f}", flush=True)
        return True

    # 运行常规上下界批次：等待 LLM 时继续整数认证，暂停新的 SA 采样，并按两侧改善提前结束。
    # 每次候选处理后下一请求重建上下文；批次结束或预算耗尽时关闭在途请求。
    def run_batch(self, searches):
        batch = self.state["batch"]
        if batch is None:
            return
        reason = "RETURN_TO_SA"
        while self.whole.remaining() > 0:
            if self.cert.converged() and not batch["validation"]:
                reason = "CONVERGED"
                break
            if self.worker is not None:
                received = self.poll(searches)
                if self.whole.remaining() <= 0:
                    break
                if received or not batch["validation"]:
                    self.cert.step()
                if not received and batch["validation"]:
                    sleep(min(.05, max(0., self.whole.remaining())))
                continue
            if not self.available():
                reason = "CALL_LIMIT"
                break
            if not batch["validation"]:
                for c in batch["channels"]:
                    if self.improved(c, batch["upper_before"], batch["lower_before"]):
                        batch["done"][c] = True
            order = ("lower", "upper") if batch["last_channel"] == "upper" else ("upper", "lower")
            channel = next((c for c in order if c in batch["channels"]
                            and not batch["done"][c] and batch["counts"][c] < 3), None)
            if channel is None:
                break
            self.start_request(channel)
        if self.whole.remaining() <= 0:
            reason = "TIME_LIMIT"
        self.close()
        batch.update(status=reason, upper_after=self.result.objective, lower_after=self.result.lower_bound,
                     sa_calls_at_end=self.whole.diagnostics["sa_calls"],
                     ended_seconds=perf_counter()-self.whole.started)
        self.state.update(phase="sa", batch=None, stalled=dict(upper=0, lower=0))

    # 按模式版本恢复停滞和批次已用额度；旧记录从新的停滞窗口迁移。
    # 原进程不能随检查点恢复，历史 RUNNING 回执标为 CANCELLED。
    def restore(self, diagnostic):
        saved = diagnostic.get("advisor_state")
        if saved is None:
            self.state["migration"] = "Legacy record: start a new SA stagnation window"
            return
        if saved["version"] != (2 if self.o.rts24_improvements else 1):
            raise ValueError("Unknown advice scheduling state version")
        self.state.clear()
        self.state.update(deepcopy(saved))
        for name in ("advisor_rounds", "llm_batches"):
            self.whole.diagnostics[name] = deepcopy(diagnostic.get(name, []))
        if self.state["batch"] is not None:
            self.state["batch"] = next(b for b in self.whole.diagnostics["llm_batches"]
                                        if b["batch_id"] == self.state["batch_id"])
        for row in self.result.llm_history:
            if row["status"] == "RUNNING":
                row.update(status="CANCELLED", message="ResumeWorkerUnavailable", accepted=0)
        self.whole.feedback = deepcopy(diagnostic.get("upper_advice_feedback"))

    # 取消在途请求、登记后台占用时间并保存上界反馈；后台时长与主线程墙钟时间可能重叠。
    def close(self):
        if self.worker is not None:
            elapsed = perf_counter()-self.worker.started
            self.whole.diagnostics["llm_background_seconds"] += elapsed
            self.pending.update(status="CANCELLED", message="SearchFinished", accepted=0,
                seconds=elapsed, background_seconds=elapsed, **self.worker.receipt)
            self.worker.close()
            self.worker = None
        self.whole.diagnostics["upper_advice_feedback"] = self.whole.feedback
