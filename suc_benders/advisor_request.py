"""One cancellable upper-advice request over HTTP or file exchange."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4
from types import SimpleNamespace as NS

from . import llm
from .data import save_json


# 单个在途请求的生命周期对象，提供非阻塞 poll 和终止/关闭操作。
class AdvisorRequest:
    # 根据 llm_agent_directory 选择文件交换或 spawn HTTP 进程，并记录总请求截止时间及请求哈希。
    # 传给子进程的数据只包含请求体和矩阵形状，避免传递求解器对象。
    def __init__(self, d, o, body, iteration, seconds):
        self.agent = None
        if o.llm_agent_directory:
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


# 管理一个独立请求目录及其 request/dispatch/reply/completion/closed 文件生命周期。
class SubagentAdvisor:
    # 生成唯一请求目录，保存请求体哈希、矩阵形状、创建时间及请求截止时间，供外部调度器读取。
    def __init__(self, d, o, body, iteration, seconds):
        self.started, self.created = perf_counter(), time()
        self.deadline, self.deadline_epoch = self.started+seconds, self.created+seconds
        self.d, self.iteration = d, iteration
        self.closed = False
        self.directory = Path(o.llm_agent_directory).resolve()/f'{iteration:02d}-{uuid4().hex}'
        self.directory.mkdir(parents=True)
        self.receipt = dict(transport='codex_subagent', request=body,
            request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            request_directory=str(self.directory))
        save_json(self.directory/'request.json', dict(request_id=self.directory.name, request=body,
            request_sha256=self.receipt['request_sha256'], shape=[d.n_units, d.horizon],
            created_epoch_seconds=self.created, deadline_epoch_seconds=self.deadline_epoch,
            reply_path=str(self.directory/'reply.json')))

    # 等待 completion 时返回 None；收到后验证请求、回复、agent、模型和推理档位，再适配公共解析器。
    # 超过原截止时间的回复丢弃；消费完成或超时后写关闭标记。
    def poll(self):
        now = perf_counter()
        if self.closed:
            return [], dict(status='CANCELLED', message='SubagentClosed', accepted=0, proposals=[], **self.receipt)
        completed = self.directory/'completion.json'
        if completed.exists():
            info = dict(status='FAILED', message='', accepted=0, format_valid=0, proposals=[])
            candidates = []
            try:
                done = json.loads(completed.read_text(encoding='utf-8'))
                dispatch = json.loads((self.directory/'dispatch.json').read_text(encoding='utf-8'))
                saved = json.loads((self.directory/'request.json').read_text(encoding='utf-8'))
                request_hash = hashlib.sha256(json.dumps(saved['request'], sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()
                text = (self.directory/'reply.json').read_text(encoding='utf-8')
                digest = hashlib.sha256(text.encode()).hexdigest()
                if (request_hash != self.receipt['request_sha256']
                        or saved['shape'] != [self.d.n_units, self.d.horizon]
                        or done['request_id'] != self.directory.name or done['request_sha256'] != self.receipt['request_sha256']
                        or done['agent_id'] != dispatch['agent_id'] or done['reply_sha256'] != digest
                        or dispatch['model'] != self.receipt['request']['model']
                        or dispatch['reasoning_effort'] != self.receipt['request']['reasoning']['effort']):
                    raise ValueError('Mismatched subagent receipt')
                # Adapt the generated text to the existing parser; this is not an HTTP response.
                raw = dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text=text)])])
                candidates, info = llm.parse_response(raw, self.d, self.iteration)
                info.update(agent_id=done['agent_id'], response_id='codex-subagent:'+done['agent_id'],
                    response_model=dispatch['model'], reasoning_effort=dispatch['reasoning_effort'],
                    response_output=[text], reply_sha256=digest,
                    seconds=done['completed_epoch_seconds']-self.created)
                if done['completed_epoch_seconds'] < self.created:
                    raise ValueError('Invalid subagent completion time')
                if done['completed_epoch_seconds'] > self.deadline_epoch:
                    candidates = []
                    info.update(status='TIME_LIMIT', accepted=0, message='LateSubagentReply')
            except (OSError, KeyError, ValueError, TypeError) as error:
                candidates = []
                info.update(status='FAILED', message=type(error).__name__, accepted=0)
            seconds = info.get('seconds', now-self.started)
            info.update(background_seconds=seconds, finished_at=self.started+seconds, **self.receipt)
            self.close()
            return candidates, info
        if now >= self.deadline:
            self.close()
            return [], dict(status='TIME_LIMIT', message='SubagentDeadline', accepted=0, format_valid=0,
                proposals=[], seconds=now-self.started, background_seconds=now-self.started, **self.receipt)
        return None

    # 标记本请求关闭并原子写入 closed.json；外部调度器负责实际子代理的停止。
    def close(self):
        self.closed = True
        if not (self.directory/'closed.json').exists():
            save_json(self.directory/'closed.json', dict(closed_epoch_seconds=time()))


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
