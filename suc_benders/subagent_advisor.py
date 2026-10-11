"""File transport for one independently dispatched Codex subagent proposal."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from . import llm


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class SubagentAdvisor:
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
                text = (self.directory/'reply.json').read_text(encoding='utf-8')
                digest = hashlib.sha256(text.encode()).hexdigest()
                if (done['request_id'] != self.directory.name or done['request_sha256'] != self.receipt['request_sha256']
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

    def close(self):
        self.closed = True
        if not (self.directory/'closed.json').exists():
            save(self.directory/'closed.json', dict(closed_epoch_seconds=time()))
