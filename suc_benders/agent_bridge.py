"""Inspect and acknowledge proposal agents dispatched by the current Codex task."""
import argparse
import hashlib
import json
from pathlib import Path
from time import time

from .subagent_advisor import save


def requests(directory):
    rows = []
    for path in sorted(Path(directory).glob('*/request.json')):
        job = json.loads(path.read_text(encoding='utf-8'))
        folder = path.parent
        dispatch = json.loads((folder/'dispatch.json').read_text()) if (folder/'dispatch.json').exists() else {}
        routing = json.loads(job['request']['input']).get('advice_routing', {})
        rows.append(dict(request_path=str(path), request_id=job['request_id'],
            remaining_seconds=max(0., job['deadline_epoch_seconds']-time()),
            closed=(folder/'closed.json').exists(), completed=(folder/'completion.json').exists(),
            reply_exists=(folder/'reply.json').exists(), agent_id=dispatch.get('agent_id'), **routing))
    return rows


def dispatch(path, agent_id, task_name):
    path = Path(path); job = json.loads(path.read_text(encoding='utf-8'))
    target = path.parent/'dispatch.json'
    if target.exists() or (path.parent/'closed.json').exists() or time() >= job['deadline_epoch_seconds']:
        raise ValueError('Request already dispatched, cancelled or expired')
    save(target, dict(agent_id=agent_id, task_name=task_name, model=job['request']['model'],
        reasoning_effort=job['request']['reasoning']['effort'], dispatched_epoch_seconds=time()))


def complete(path, agent_id):
    path = Path(path); job = json.loads(path.read_text(encoding='utf-8'))
    recorded = json.loads((path.parent/'dispatch.json').read_text())
    if recorded['agent_id'] != agent_id or (path.parent/'completion.json').exists():
        raise ValueError('Completion agent mismatch or duplicate completion')
    reply = path.parent/'reply.json'
    save(path.parent/'completion.json', dict(agent_id=agent_id, request_id=job['request_id'],
        request_sha256=job['request_sha256'], reply_sha256=hashlib.sha256(reply.read_bytes()).hexdigest(), completed_epoch_seconds=time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--agent-id')
    parser.add_argument('--task-name')
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    if args.request:
        if args.complete:
            complete(args.request, args.agent_id)
        else:
            dispatch(args.request, args.agent_id, args.task_name)
    else:
        print(json.dumps(requests(args.directory), ensure_ascii=False))


if __name__ == '__main__':
    main()
