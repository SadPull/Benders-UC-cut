import json
from time import time

import pytest

from benchmarks.agent_bridge import complete, dispatch, requests
from suc_benders.bound_advisor import BoundAdvisor
from suc_benders.config import load_options
from suc_benders.subagent_advisor import save


def task(tmp_path, tiny, options, seconds=30):
    options.llm_agent_directory = str(tmp_path)
    options.llm_model = 'gpt-6-astra'
    options.llm_api_key = 'must-not-be-exported'
    body = dict(model=options.llm_model, reasoning=dict(effort='max'), input=json.dumps(dict(advice_routing=dict(channel='upper'))))
    return BoundAdvisor(tiny(1)[0], options, body, 1, seconds)


def test_subagent_reply_requires_dispatch_and_completion_receipts(tmp_path, tiny, options, monkeypatch):
    monkeypatch.setattr('suc_benders.bound_advisor.mp.get_context', lambda *a: pytest.fail('HTTP process started'))
    advisor = task(tmp_path, tiny, options)
    folder = advisor.agent.directory
    request = folder/'request.json'
    assert len(requests(tmp_path)) == 1 and requests(tmp_path)[0]['agent_id'] is None
    assert 'must-not-be-exported' not in request.read_text()
    dispatch(request, 'real-agent-id', 'upper')
    save(folder/'reply.json', dict(candidates=[dict(u=[[1, 1, 1]])]))
    assert advisor.poll() is None
    complete(request, 'real-agent-id')
    candidates, receipt = advisor.poll()
    assert receipt['status'] == 'OK' and len(candidates) == 1
    assert receipt['transport'] == 'codex_subagent' and receipt['agent_id'] == 'real-agent-id'
    assert receipt['response_model'] == 'gpt-6-astra' and receipt['reasoning_effort'] == 'max'
    assert receipt['reply_sha256'] and receipt['request_sha256']
    assert requests(tmp_path)[0]['closed']


def test_expired_request_is_closed_and_cannot_be_dispatched(tmp_path, tiny, options):
    advisor = task(tmp_path, tiny, options, 0.)
    candidates, receipt = advisor.poll()
    assert not candidates and receipt['status'] == 'TIME_LIMIT'
    with pytest.raises(ValueError, match='expired'):
        dispatch(advisor.agent.directory/'request.json', 'late-agent', 'late')


@pytest.mark.parametrize('alter', ['late', 'hash', 'agent'])
def test_late_or_unrelated_completion_cannot_supply_candidates(tmp_path, tiny, options, alter):
    advisor = task(tmp_path, tiny, options)
    folder = advisor.agent.directory
    dispatch(folder/'request.json', 'agent', 'upper')
    save(folder/'reply.json', dict(candidates=[dict(u=[[1, 1, 1]])]))
    complete(folder/'request.json', 'agent')
    done = json.loads((folder/'completion.json').read_text())
    done.update({'completed_epoch_seconds':time()+100} if alter == 'late' else
                {'request_sha256':'wrong'} if alter == 'hash' else {'agent_id':'other-agent'})
    save(folder/'completion.json', done)
    candidates, receipt = advisor.poll()
    assert not candidates and receipt['status'] in ('TIME_LIMIT', 'FAILED')


def test_subagent_configuration_does_not_require_http_credentials(tmp_path):
    options = load_options(dict(llm_enabled=True, llm_model='gpt-6-astra', llm_agent_directory=str(tmp_path)), tmp_path/'absent.json')
    assert not options.llm_api_key and not options.llm_url


def test_cancelled_request_cannot_consume_a_later_reply(tmp_path, tiny, options):
    advisor = task(tmp_path, tiny, options)
    folder = advisor.agent.directory
    dispatch(folder/'request.json', 'agent', 'upper')
    advisor.close()
    save(folder/'reply.json', dict(candidates=[dict(u=[[1, 1, 1]])]))
    complete(folder/'request.json', 'agent')
    candidates, receipt = advisor.poll()
    assert not candidates and receipt['status'] == 'CANCELLED'


def test_timely_reply_can_be_polled_after_request_deadline(tmp_path, tiny, options, monkeypatch):
    advisor = task(tmp_path, tiny, options)
    folder = advisor.agent.directory
    dispatch(folder/'request.json', 'agent', 'upper')
    save(folder/'reply.json', dict(candidates=[dict(u=[[1, 1, 1]])]))
    complete(folder/'request.json', 'agent')
    monkeypatch.setattr('suc_benders.subagent_advisor.perf_counter', lambda: advisor.deadline+1)
    candidates, receipt = advisor.poll()
    assert len(candidates) == 1 and receipt['status'] == 'OK'
