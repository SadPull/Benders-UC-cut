import hashlib
import json
import os
import subprocess
import sys
from time import perf_counter, sleep

import pytest

from suc_benders import advisor_request
from suc_benders.config import load_options
from conftest import whole_day


@pytest.fixture
def fake_codex(tmp_path):
    folder = tmp_path / 'fake codex with spaces'
    folder.mkdir()
    script = folder / 'fake.py'
    script.write_text('''import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
folder = Path(args[args.index('--output-last-message')+1]).parent
settings = json.loads(Path(__file__).with_name('settings.json').read_text())
(folder/'stdin.txt').write_bytes(sys.stdin.buffer.read())
(folder/'observed.json').write_text(json.dumps(dict(argv=args, pid=os.getpid(),
    has_codex_api_key='CODEX_API_KEY' in os.environ)))
print(json.dumps(dict(type='thread.started', thread_id='fake-thread')), flush=True)
if settings['mode'] == 'tree':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (folder/'child-pid.txt').write_text(str(child.pid))
    time.sleep(60)
elif settings['mode'] != 'missing':
    text = '{' if settings['mode'] == 'invalid_json' else json.dumps(settings['reply'])
    (folder/'reply.json').write_text(text, encoding='utf-8')
print(json.dumps(dict(type='turn.completed', usage=dict(input_tokens=10, output_tokens=5))), flush=True)
sys.exit(23 if settings['mode'] == 'nonzero' else 0)
''', encoding='utf-8')
    command = folder / ('codex.cmd' if os.name == 'nt' else 'codex')
    command.write_text((f'@echo off\n"{sys.executable}" "{script}" %*\n' if os.name == 'nt' else
                        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n'), encoding='utf-8')
    command.chmod(0o755)

    def make(mode='valid', reply=None):
        (folder/'settings.json').write_text(json.dumps(dict(mode=mode, reply=reply if reply is not None else
            dict(candidates=[dict(name='test', reason='test', u=[[1, 1, 1]])]))), encoding='utf-8')
        return str(command)

    return make


def start(tiny, options, tmp_path, command, seconds=10.):
    options.llm_codex_command = command
    options.llm_agent_directory = str(tmp_path / 'request evidence with spaces')
    options.llm_api_key = 'must-not-be-exported'
    body = dict(model='gpt-6-astra', reasoning=dict(effort='max'), instructions='只返回候选 JSON。',
                input=json.dumps(dict(message='中文 & "$()` 提示'), ensure_ascii=False))
    return advisor_request.AdvisorRequest(tiny(1)[0], options, body, 1, seconds)


def finish(task):
    try:
        deadline = perf_counter()+12.
        while perf_counter() < deadline:
            reply = task.poll()
            if reply is not None:
                return reply
            sleep(.01)
        pytest.fail('Codex worker did not finish')
    finally:
        task.close()


def test_codex_configuration_without_http_credentials(tmp_path):
    options = load_options(dict(llm_enabled=True, llm_model='gpt-6-astra',
        llm_codex_command='codex', llm_agent_directory=str(tmp_path)), tmp_path/'absent.json')
    assert not options.llm_url and not options.llm_api_key


@pytest.mark.parametrize('setting', [None, 12])
def test_codex_command_must_be_a_string(tmp_path, setting):
    with pytest.raises(ValueError, match='llm_codex_command'):
        load_options(dict(llm_codex_command=setting), tmp_path/'absent.json')


def test_codex_requires_evidence_directory(tmp_path):
    with pytest.raises(ValueError, match='llm_agent_directory'):
        load_options(dict(llm_codex_command='codex'), tmp_path/'absent.json')


def test_https_override_preserves_custom_provider_config(tmp_path):
    config = tmp_path/'config.toml'
    original = 'model_provider="proxy"\n[model_providers.proxy]\nname="local"\nsupports_websockets=true\n'
    config.write_text(original, encoding='utf-8')
    assert advisor_request._codex_https_args(config) == ['-c', 'model_providers.proxy.supports_websockets=false']
    assert config.read_text(encoding='utf-8') == original
    defaults = advisor_request._codex_https_args(tmp_path/'absent.toml')
    assert 'model_providers.openai.supports_websockets=false' not in defaults
    assert any('supports_websockets=false' in arg for arg in defaults)


def test_partial_event_log_still_supplies_thread_and_redacted_error(tmp_path):
    (tmp_path/'events.jsonl').write_text(
        '{"type":"thread.started","thread_id":"live-thread"}\n'
        '{"type":"error","message":"401 invalid_api_key sk-private-secret"}\n'
        '[]\n'
        '{"type":"unfinished', encoding='utf-8')
    metadata = advisor_request._codex_event_info(tmp_path)
    assert metadata['thread_id'] == 'live-thread'
    assert '401' in metadata['last_cli_error'] and 'sk-private-secret' not in metadata['last_cli_error']


def test_failed_worker_start_still_saves_receipt(tiny, options, tmp_path, fake_codex, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('cannot start worker')

    monkeypatch.setattr('multiprocessing.process.BaseProcess.start', fail)
    with pytest.raises(OSError, match='cannot start worker'):
        start(tiny, options, tmp_path, fake_codex())
    receipt, = (tmp_path/'request evidence with spaces').glob('*/receipt.json')
    info = json.loads(receipt.read_text(encoding='utf-8'))
    assert info['status'] == 'FAILED' and info['exit_code'] is None


def test_codex_stdin_schema_receipt_and_fresh_sessions(tiny, options, tmp_path, fake_codex, monkeypatch):
    monkeypatch.setenv('CODEX_API_KEY', 'parent-agent-only-key')
    folders = []
    for _ in range(2):
        task = start(tiny, options, tmp_path, fake_codex())
        candidates, info = finish(task)
        folder = task.codex_directory
        folders.append(folder)
        assert len(candidates) == 1 and info['status'] == 'OK'
        assert info['transport'] == 'codex_exec' and info['exit_code'] == 0
        assert info['thread_id'] == 'fake-thread' and info['usage']['input_tokens'] == 10
        assert info['requested_model'] == 'gpt-6-astra' and info['reasoning_effort'] == 'max'
        prompt = (folder/'prompt.txt').read_bytes()
        assert prompt == (folder/'stdin.txt').read_bytes()
        assert json.loads(prompt.decode('utf-8').split('\n\n')[1])['message'] == '中文 & "$()` 提示'
        assert info['prompt_sha256'] == hashlib.sha256(prompt).hexdigest()
        assert info['reply_sha256'] == hashlib.sha256((folder/'reply.json').read_bytes()).hexdigest()
        saved = json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
        assert saved['status'] == 'OK' and saved['request_sha256'] == info['request_sha256']
        assert 'must-not-be-exported' not in json.dumps(saved)
        schema = json.loads((folder/'schema.json').read_text())
        matrix = schema['properties']['candidates']['items']['properties']['u']
        assert matrix['minItems'] == matrix['maxItems'] == 1
        assert matrix['items']['minItems'] == matrix['items']['maxItems'] == 3
        observed = json.loads((folder/'observed.json').read_text())
        assert not observed['has_codex_api_key']
        assert os.environ['CODEX_API_KEY'] == 'parent-agent-only-key'
        args = observed['argv']
        assert any('supports_websockets=false' in arg for arg in args)
        assert '--sandbox' in args and 'read-only' in args and args[-1] == '-'
        assert 'agents.enabled=false' in args and 'approval_policy="never"' in args
        assert 'model_reasoning_effort="max"' in args and 'resume' not in args
    assert folders[0] != folders[1]


@pytest.mark.parametrize('mode,reply,status', [
    ('valid', dict(candidates=[]), 'OK'),
    ('valid', dict(candidates=[dict(u=[[1, 1]])]), 'FAILED'),
    ('valid', dict(candidates=[dict(u=[[1, .5, 1]])]), 'FAILED'),
    ('valid', dict(candidates=[dict(u=[[1, 1, 1]])]*5), 'FAILED'),
    ('invalid_json', None, 'FAILED'),
    ('missing', None, 'FAILED'),
    ('nonzero', None, 'FAILED'),
])
def test_codex_invalid_or_empty_results(tiny, options, tmp_path, fake_codex, mode, reply, status):
    task = start(tiny, options, tmp_path, fake_codex(mode, reply))
    candidates, info = finish(task)
    assert not candidates and info['status'] == status
    assert info['exit_code'] == (23 if mode == 'nonzero' else 0)
    assert json.loads((task.codex_directory/'receipt.json').read_text(encoding='utf-8'))['status'] == status


def test_missing_codex_is_recorded_without_http_fallback(tiny, options, tmp_path):
    task = start(tiny, options, tmp_path, str(tmp_path/'absent-codex'))
    candidates, info = finish(task)
    assert not candidates and info['status'] == 'FAILED'
    assert 'FileNotFoundError' in info['message'] and info['exit_code'] is None
    assert (task.codex_directory/'request.json').exists()


@pytest.mark.parametrize('cancel', [False, True])
def test_codex_timeout_or_close_kills_owned_descendants(tiny, options, tmp_path, fake_codex, cancel):
    task = start(tiny, options, tmp_path, fake_codex('tree'), seconds=10. if cancel else 3.)
    child_path = task.codex_directory/'child-pid.txt'
    try:
        deadline = perf_counter()+8.
        while not child_path.exists() and perf_counter() < deadline:
            sleep(.01)
        assert child_path.exists()
        if cancel:
            task.close()
            candidates, info = task.poll()
        else:
            candidates, info = finish(task)
        assert not candidates and info['status'] == ('CANCELLED' if cancel else 'TIME_LIMIT')
        assert not task.process.is_alive()
        cli_pid = json.loads((task.codex_directory/'observed.json').read_text())['pid']
        for pid in (int(child_path.read_text()), cli_pid):
            if os.name == 'nt':
                listing = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                    capture_output=True, check=True).stdout
                assert f'"{pid}"'.encode() not in listing
            else:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
        saved = json.loads((task.codex_directory/'receipt.json').read_text(encoding='utf-8'))
        assert saved['status'] == info['status'] and saved['thread_id'] == 'fake-thread'
    finally:
        task.close()


def test_completed_reply_can_be_polled_after_deadline(tiny, options, tmp_path, fake_codex, monkeypatch):
    task = start(tiny, options, tmp_path, fake_codex())
    try:
        task.process.join(timeout=8.)
        assert not task.process.is_alive()
        monkeypatch.setattr(advisor_request, 'perf_counter', lambda: task.deadline+1.)
        candidates, info = task.poll()
        assert len(candidates) == 1 and info['status'] == 'OK'
    finally:
        task.close()


def test_codex_backend_reaches_numeric_validation_after_stagnation(tiny, options, tmp_path, fake_codex, monkeypatch):
    whole = whole_day(tiny, options)
    options.llm_enabled = True
    options.llm_model = 'gpt-6-astra'
    options.llm_codex_command = fake_codex(reply=dict(candidates=[
        dict(name='one-unit', reason='lower fixed cost', u=[[0, 0, 0], [1, 1, 1]])]))
    options.llm_agent_directory = str(tmp_path/'llm_requests')
    options.llm_max_calls = 1
    monkeypatch.setattr(whole.certifier, 'step', lambda: sleep(.01))
    monkeypatch.setattr(whole.certifier, 'converged', lambda: False)
    for number in range(1, 4):
        before = whole.diagnostics['sa_calls']
        whole.diagnostics['sa_calls'] += 1
        whole.advice.after_round(whole.result.objective, before, [], 1, number)
    assert whole.result.llm_calls == 1
    assert whole.result.objective == pytest.approx(225.)
    assert whole.result.validation['feasible']
    receipt, = (tmp_path/'llm_requests').glob('*/receipt.json')
    assert json.loads(receipt.read_text(encoding='utf-8'))['status'] == 'OK'
    feedback = whole.result.llm_history[0]['feedback']['proposals'][0]
    assert feedback['validation_completed'] and feedback['refreshed_incumbent']
