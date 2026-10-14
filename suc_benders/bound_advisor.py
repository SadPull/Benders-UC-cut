"""One cancellable Responses request; optimization stays in the parent process."""
# 模块职责：封装可取消的单次 LLM 请求，可使用后台 HTTP 进程或文件交换子代理。
# 工作进程只处理传输和候选解析，求解模型、割计算和费用验证始终留在父进程。
import hashlib
import json
import multiprocessing as mp
from time import perf_counter
from types import SimpleNamespace as NS

from . import llm


# 在公共提示词上构建下界探测请求，引导候选揭示主问题低估或调度不可行区域。
# LLM 只提议 u 矩阵，不能提供未经数值证明的费用、界或割系数。
def request_body(d, scenarios, best, o, context, iteration):
    body = llm.request_body(d, scenarios, best, o, {"bound_search": context}, [], iteration)
    body["instructions"] = (
        "You advise a certified integer Benders master for stochastic unit commitment. "
        "Propose binary u schedules to probe underestimated dispatch costs or missing "
        "dispatch feasibility restrictions, not only to improve the incumbent. "
        "Use the current master schedule, its predicted and evaluated costs, and cut feedback. "
        "Every probe is evaluated numerically; never invent costs, bounds, dispatch or cut coefficients. "
        "Respect commitment transitions, initial conditions, minimum up/down times and terminal rules. "
        "Dispatch-infeasible probes can be useful and will generate feasibility cuts. "
        "No heuristic repair is applied. Return JSON with at most four candidates, each with "
        "name, reason and an exact units-by-horizon 0/1 u matrix."
    )
    prompt = json.loads(body["input"])
    if o.rts24_improvements:
        body['instructions'] = body['instructions'].replace('at most four candidates', 'at most two candidates')
    prompt["task"] = "Suggest diverse probes that strengthen the certified master lower bound."
    prompt["terminal_rule"] = ("Shutdown in the last hour is allowed; minimum times truncate at the horizon."
        if getattr(d, "model_profile", "") == "azema_sms" else
        "Shutdown in the last hour is forbidden; minimum times truncate at the horizon.")
    prompt["rules"] = ["Return exact binary u matrices and JSON only", "Do not repeat previous probes",
                       "Prioritize underestimated master solutions and informative nearby alternatives",
                       "Respect discrete commitment rules; dispatch feasibility is checked by the solver"]
    body["input"] = json.dumps(prompt, separators=(",", ":"), allow_nan=False)
    return body


# 后台进程入口：发送请求、解析候选，并通过管道返回脱敏回执和完成时刻。
# 异常只保留类型及 HTTP 状态码；无论发送结果是否成功，最后都关闭管道发送端。
def _request(connection, url, key, body, shape, iteration, seconds):
    started = perf_counter()
    candidates = []
    info = dict(status="FAILED", message="", accepted=0, format_valid=0, proposals=[])
    try:
        raw = llm.post_response(url, key, body, seconds)
        candidates, info = llm.parse_response(raw, shape, iteration)
        info.update(response_id=raw.get("id"), response_model=raw.get("model"), usage=raw.get("usage"),
                    response_output=[part["text"] for item in raw.get("output", [])
                                     if item.get("type") == "message" for part in item.get("content", [])
                                     if part.get("type") == "output_text"])
    except Exception as error:
        info["message"] = type(error).__name__
        if isinstance(getattr(error, "code", None), int):
            info["http_status"] = error.code
    finished = perf_counter()
    info.update(seconds=finished-started, finished_at=finished)
    try:
        connection.send((candidates, info))
    finally:
        connection.close()


# 单个在途请求的生命周期对象，提供非阻塞 poll 和终止/关闭操作。
class BoundAdvisor:
    # 根据 llm_agent_directory 选择文件交换或 spawn HTTP 进程，并记录总请求截止时间及请求哈希。
    # 传给子进程的数据只包含请求体和矩阵形状，避免传递求解器对象。
    def __init__(self, d, o, body, iteration, seconds):
        self.agent = None
        if o.llm_agent_directory:
            from .subagent_advisor import SubagentAdvisor
            self.agent = SubagentAdvisor(d, o, body, iteration, seconds)
            self.started, self.deadline, self.receipt = self.agent.started, self.agent.deadline, self.agent.receipt
            return
        self.started = perf_counter()
        self.deadline = self.started + seconds
        context = mp.get_context("spawn")
        self.connection, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=_request,
            args=(sender, o.llm_url, o.llm_api_key, body,
                  NS(n_units=d.n_units, horizon=d.horizon), iteration, seconds), daemon=True)
        self.receipt = dict(request=body, request_sha256=hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        try:
            self.process.start()
        except Exception:
            self.connection.close()
            raise
        finally:
            # 父进程释放自己的发送端副本，保证子进程结束后读端能正确观察到 EOF。
            sender.close()

    # 尚未完成时返回 None；完成、超时或进程退出时返回 (候选, 回执) 并回收资源。
    # 完成时刻晚于请求截止时间的回复必须丢弃，即使当前已经可以读到结果。
    def poll(self):
        if self.agent is not None:
            return self.agent.poll()
        now = perf_counter()
        if self.connection.poll():
            try:
                candidates, info = self.connection.recv()
            except EOFError:
                candidates, info = [], dict(status="FAILED", message="WorkerExited", accepted=0, proposals=[])
            if info.get("finished_at", now) > self.deadline:
                candidates = []
                info.update(status="TIME_LIMIT", accepted=0)
            info["background_seconds"] = max(0., info.get("finished_at", now)-self.started)
            info.update(self.receipt)
            self.close()
            return candidates, info
        if now >= self.deadline or not self.process.is_alive():
            info = dict(status="TIME_LIMIT" if now >= self.deadline else "FAILED",
                        message="RequestDeadline" if now >= self.deadline else "WorkerExited",
                        seconds=now-self.started, background_seconds=now-self.started,
                        accepted=0, proposals=[], **self.receipt)
            self.close()
            return [], info
        return None

    # 关闭文件交换请求，或先终止再按需强制杀死 HTTP 工作进程，最后回收管道。
    def close(self):
        if self.agent is not None:
            self.agent.close()
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)
        self.connection.close()
