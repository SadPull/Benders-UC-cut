"""File transport for one independently dispatched Codex subagent proposal."""
# 模块职责：为单个外部子代理请求提供原子 JSON 文件交换、身份/哈希核对和截止时间处理。
# 单调时钟用于本进程计时，Unix 时间戳用于跨进程回执；本模块不自行派发外部子代理。
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from . import llm


# 先写同目录临时文件并 flush/fsync，再原子替换目标 JSON，减少检查点半写和断电丢失风险。
def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


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
        save(self.directory/'request.json', dict(request_id=self.directory.name, request=body,
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
                if done['completed_epoch_seconds'] > self.deadline_epoch:
                    candidates = []
                    info.update(status='TIME_LIMIT', accepted=0, message='LateSubagentReply')
            except (OSError, KeyError, ValueError, TypeError) as error:
                info['message'] = type(error).__name__
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
            save(self.directory/'closed.json', dict(closed_epoch_seconds=time()))
