"""Inspect and acknowledge proposal agents dispatched by the current Codex task."""
# 模块职责：读取文件交换请求，登记外部子代理身份，并记录回复完成凭据。
# 这里只管理 request/dispatch/reply/completion 文件；外部调度器负责启动实际子代理。
import argparse
import hashlib
import json
from pathlib import Path
from time import time

from .data import save_json


# 扫描请求目录，返回每个请求的剩余秒数、调度身份和文件完成状态。
# 剩余时间使用 Unix 时间戳计算，列表查询不会启动或续跑任务。
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


# 为单个 request.json 登记 agent_id、任务名、模型和推理档位。
# 已登记、已关闭或已过期的请求不能再次派发。
def dispatch(path, agent_id, task_name):
    path = Path(path); job = json.loads(path.read_text(encoding='utf-8'))
    target = path.parent/'dispatch.json'
    if target.exists() or (path.parent/'closed.json').exists() or time() >= job['deadline_epoch_seconds']:
        raise ValueError('Request already dispatched, cancelled or expired')
    save_json(target, dict(agent_id=agent_id, task_name=task_name, model=job['request']['model'],
        reasoning_effort=job['request']['reasoning']['effort'], dispatched_epoch_seconds=time()))


# 将完成通知绑定到已登记的 agent_id，并保存请求身份和回复 SHA256。
# 消费方还会验证这些哈希与截止时间，写入完成通知本身不表示候选已通过数值验证。
def complete(path, agent_id):
    path = Path(path); job = json.loads(path.read_text(encoding='utf-8'))
    recorded = json.loads((path.parent/'dispatch.json').read_text())
    if recorded['agent_id'] != agent_id or (path.parent/'completion.json').exists():
        raise ValueError('Completion agent mismatch or duplicate completion')
    reply = path.parent/'reply.json'
    save_json(path.parent/'completion.json', dict(agent_id=agent_id, request_id=job['request_id'],
        request_sha256=job['request_sha256'], reply_sha256=hashlib.sha256(reply.read_bytes()).hexdigest(), completed_epoch_seconds=time()))


# 命令行入口：按参数查询请求、登记派发，或写入完成通知。
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
