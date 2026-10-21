import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from conftest import whole_day
from suc_benders import advisor_request as transport

SESSION = '01a15229-5327-7e13-af26-d1281e24d0bb'
OTHER = '01a15235-74b6-7352-bba9-6f7dc1008463'


@pytest.fixture
def cli_environment(tmp_path, monkeypatch):
    folder = tmp_path/'sessions'; folder.mkdir()
    (folder/f'rollout-{SESSION}.jsonl').write_text('{}\n', encoding='utf-8')
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.setattr(transport.shutil, 'which', lambda command: sys.executable)
    monkeypatch.setattr(transport.signal, 'signal', lambda *args: None)
    return dict(instructions='Only return candidates', input='{"upper":100}',
                model='gpt-6-astra', reasoning={'effort': 'max'})


def test_resume_command_has_explicit_identity_and_current_context(tmp_path, cli_environment):
    args, _, _ = transport._prepare_codex('codex', cli_environment, NS(n_units=1, horizon=1), tmp_path, SESSION)
    assert args[args.index('resume')+1] == SESSION and '--last' not in args
    assert args[args.index('--sandbox')+1] == 'read-only'
    assert args[args.index('--model')+1] == 'gpt-6-astra'
    assert '--output-schema' in args
    prompt = (tmp_path/'prompt.txt').read_text(encoding='utf-8')
    assert 'supersede older states' in prompt and '"upper":100' in prompt


def test_delta_prompt_without_instructions_starts_with_continuation(tmp_path, cli_environment):
    body = dict(cli_environment, instructions='')
    transport._prepare_codex('codex', body, NS(n_units=1, horizon=1), tmp_path, SESSION)
    prompt = (tmp_path/'prompt.txt').read_text(encoding='utf-8')
    assert prompt.startswith('Continue this conversation.') and '\n\n\n' not in prompt
    assert 'Only return candidates' not in prompt and '"upper":100' in prompt


@pytest.mark.parametrize('event_id,timeout,expected', [(SESSION, False, 'OK'),
    (OTHER, False, 'SESSION_FAILED'), (SESSION, True, 'TIME_LIMIT')])
def test_cli_reply_identity_and_timeout(tmp_path, cli_environment, monkeypatch, event_id, timeout, expected):
    sent = []
    def popen(args, **kwargs):
        kwargs['stdout'].write((json.dumps(dict(type='thread.started', thread_id=event_id))+'\n').encode())
        kwargs['stdout'].write(b'{"type":"turn.completed","usage":{}}\n')
        Path(args[args.index('--output-last-message')+1]).write_text(
            json.dumps({"candidates": [{"name": "one", "reason": "test", "changes": [
                {"unit": 1, "on_intervals": [{"start": 1, "end": 1}]}]}]}), encoding='utf-8')
        def wait(**kw):
            if timeout:
                raise subprocess.TimeoutExpired(args, 20)
            return 0
        return NS(pid=123, wait=wait, poll=lambda: 0)
    monkeypatch.setattr(transport.subprocess, 'Popen', popen)
    transport._request_codex(NS(send=sent.append, close=lambda: None), 'codex', cli_environment,
        NS(n_units=1, horizon=1, incumbent_u=[[0]]), 4, perf_counter()+20, tmp_path, SESSION)
    candidates, info = sent[0]
    assert info['status'] == expected
    assert len(candidates) == (1 if expected == 'OK' else 0)
    if expected == 'OK':
        assert candidates[0][0].tolist() == [[1.]] and candidates[0][1]['raw_changes'][0]['unit'] == 1


def test_missing_session_does_not_start_a_new_conversation(tmp_path, cli_environment, monkeypatch):
    sent = []
    monkeypatch.setattr(transport.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected CLI launch'))
    transport._request_codex(NS(send=sent.append, close=lambda: None), 'codex', cli_environment,
        NS(n_units=1, horizon=1), 4, perf_counter()+20, tmp_path, OTHER)
    assert sent[0][1]['status'] == 'SESSION_FAILED' and not sent[0][0]


def setup_advice(tiny, options, monkeypatch):
    whole = whole_day(tiny, options)
    options.llm_enabled = True; options.llm_model = 'gpt-6-astra'; options.llm_codex_command = 'codex'
    options.llm_timeout = 1800; options.llm_reserve_seconds = 60; options.llm_max_calls = None
    whole.deadline = perf_counter()+7200
    whole.result.llm_calls = 3
    whole.result.llm_history = [dict(status='OK', thread_id=SESSION, feedback={'previous': 2}),
                                dict(status='TIME_LIMIT', thread_id=OTHER)]
    whole.feedback = {'previous': 2}
    whole.advice.thread_id = SESSION
    monkeypatch.setattr(whole, 'upper_context', lambda: (
        {'upper': whole.result.objective, 'previous_call_feedback': whole.feedback}, [], None))
    monkeypatch.setattr('suc_benders.upper_advisor.llm.request_body',
        lambda d, s, r, o, context, bases, iteration, continuation=False:
            dict(instructions='' if continuation else 'test', input=json.dumps(dict(context, delta=continuation,
                                     incumbent_u=whole.result.u.tolist())),
                 model=o.llm_model, reasoning={'effort': 'max'}))
    def consume(proposals, info, searches, baseline):
        info['feedback'] = {'measured_upper': whole.result.objective}
        whole.feedback = info['feedback']
    monkeypatch.setattr(whole, 'consume_upper', consume)
    return whole


def test_multiple_calls_keep_session_and_send_new_measured_feedback(tiny, options, monkeypatch):
    whole = setup_advice(tiny, options, monkeypatch)
    calls = []
    class Worker:
        def __init__(self, d, o, body, iteration, seconds, *, session_id):
            calls.append((iteration, seconds, session_id, json.loads(body['input'])))
        def poll(self):
            return [], dict(status='OK', thread_id=SESSION, background_seconds=1., proposals=[])
    monkeypatch.setattr(transport, 'AdvisorRequest', Worker)
    whole.advice.begin()
    whole.advice.start_request(); whole.advice.start_request()
    assert len(calls) == 1 and whole.result.llm_calls == 4
    # 回执记下模型看到的基准方案，增量候选据此还原。
    assert whole.result.llm_history[-1]['base_u'] == whole.result.u.tolist()
    whole.advice.poll([])
    prior_upper = whole.result.objective
    whole.result.objective -= 1
    whole.advice.start_request(); whole.advice.poll([])
    assert [r[:3] for r in calls] == [(4, 1800, SESSION), (5, 1800, SESSION)]
    assert calls[0][3]['previous_call_feedback'] == {'previous': 2}
    assert calls[1][3]['previous_call_feedback'] == {'measured_upper': prior_upper}
    assert 'last_successful_call_feedback' not in calls[1][3]
    assert calls[1][3]['upper'] == prior_upper-1
    assert 'advice_routing' not in calls[1][3] and whole.result.llm_history[-1]['batch_call'] == 2


def test_short_window_never_starts_or_charges_a_call(tiny, options, monkeypatch):
    whole = setup_advice(tiny, options, monkeypatch)
    whole.deadline = perf_counter()+1859
    monkeypatch.setattr(transport, 'AdvisorRequest', lambda *a, **k: pytest.fail('Short-window call'))
    whole.advice.begin(); batch = whole.advice.state['batch']
    failures = whole.advice.state['consecutive_failures']
    whole.advice.advance([])
    assert batch['status'] == 'LLM_WINDOW_TOO_SHORT'
    assert whole.result.llm_calls == 3 and whole.advice.state['consecutive_failures'] == failures


def test_inflight_call_keeps_polling_and_convergence_cancels_it(tiny, options, monkeypatch):
    whole = setup_advice(tiny, options, monkeypatch)
    closed = []
    whole.advice.begin()
    whole.advice.pending = {'status': 'RUNNING'}
    whole.advice.worker = NS(poll=lambda: None, close=lambda: closed.append(True), started=perf_counter(), receipt={})
    whole.deadline = perf_counter()+50
    whole.advice.advance([])
    assert not closed and whole.advice.worker is not None
    monkeypatch.setattr(whole.certifier, 'converged', lambda: True)
    whole.advice.advance([])
    assert closed == [True] and whole.advice.worker is None


def test_session_mismatch_resets_session_without_evaluating_candidates(tiny, options, monkeypatch):
    whole = setup_advice(tiny, options, monkeypatch)
    whole.advice.begin(); whole.advice.pending = {'status': 'RUNNING'}
    whole.advice.worker = NS(poll=lambda: ([object()], dict(status='OK', thread_id=OTHER,
                                                          background_seconds=1., proposals=[])))
    monkeypatch.setattr(whole, 'consume_upper', lambda *a: pytest.fail('Wrong-session candidate evaluated'))
    whole.advice.poll([])
    assert whole.advice.pending['status'] == 'SESSION_FAILED'
    assert whole.advice.thread_id is None and whole.advice.state['session_resets'] == 1
    assert not whole.advice.state['disabled']


def test_failure_after_session_reset_disables_llm_and_keeps_searching(tiny, options, monkeypatch):
    whole = setup_advice(tiny, options, monkeypatch)
    calls = []
    deltas = []
    class Worker:
        def __init__(self, d, o, body, iteration, seconds, *, session_id):
            calls.append(session_id); deltas.append(json.loads(body['input'])['delta'])
        def poll(self):
            return [], dict(status='FAILED', message='boom', background_seconds=1., proposals=[])
    monkeypatch.setattr(transport, 'AdvisorRequest', Worker)
    batches = []
    for _ in range(8):
        if whole.advice.state['batch'] is None and whole.advice.available():
            whole.advice.begin(); batches.append(whole.advice.state['batch'])
        whole.advice.advance([])
    # 旧会话连续两次失败 → 新会话；新会话再连续两次失败（跨批次）→ 停用，不再计费。
    assert calls == [SESSION, SESSION, None, None]
    # 续接旧会话发增量；会话重置后新会话必须发完整提示。
    assert deltas == [True, True, False, False]
    assert [r['prompt_mode'] for r in whole.result.llm_history[2:]] == ['delta', 'delta', 'full', 'full']
    assert [b['status'] for b in batches] == ['BATCH_LIMIT', 'LLM_DISABLED']
    assert whole.advice.state['disabled'] and whole.diagnostics['llm_disabled']['reason']
    assert whole.advice.state['batch'] is None and not whole.advice.available()
    assert whole.result.llm_calls == 7


@pytest.mark.parametrize('enabled', [True, False])
def test_kickoff_starts_one_request_only_when_enabled(tiny, options, monkeypatch, enabled):
    whole = setup_advice(tiny, options, monkeypatch)
    whole.result.llm_calls = 0; whole.result.llm_history = []
    options.llm_first_call_at_start = enabled
    calls = []
    class Worker:
        def __init__(self, d, o, body, iteration, seconds, *, session_id):
            calls.append(iteration)
        def poll(self):
            return None
    monkeypatch.setattr(transport, 'AdvisorRequest', Worker)
    whole.advice.kickoff()
    whole.advice.advance([]); whole.advice.advance([])
    batch = whole.advice.state['batch']
    if enabled:
        assert batch['trigger'] == 'start' and calls == [1]
    else:
        assert batch is None and calls == []
