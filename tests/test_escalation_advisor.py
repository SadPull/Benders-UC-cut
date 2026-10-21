import json
import sys
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from conftest import whole_day
from suc_benders import advisor_request as transport
from suc_benders.config import load_options

SESSION = '01a15229-5327-7e13-af26-d1281e24d0bb'


@pytest.fixture
def lanes(tiny, options, monkeypatch, tmp_path):
    whole = whole_day(tiny, options)
    options.llm_enabled = True; options.llm_model = 'deepseek-v4.1-flash'
    options.llm_timeout = 600; options.llm_reserve_seconds = 60; options.llm_max_calls = None
    options.llm_escalation_model = 'gpt-6-astra'; options.llm_escalation_command = 'codex'
    options.llm_escalation_directory = str(tmp_path/'requests'); options.llm_escalation_timeout = 1800
    options.llm_escalation_min_seconds = 700; options.llm_escalation_gain = .005
    whole.deadline = perf_counter()+7200
    drops, replies, workers = {}, {}, []
    monkeypatch.setattr(whole, 'upper_context', lambda: ({'upper': whole.result.objective}, [], None))
    monkeypatch.setattr('suc_benders.upper_advisor.llm.request_body',
        lambda d, s, r, o, context, bases, iteration, continuation=False:
            dict(instructions='test', model=o.llm_model, reasoning={'effort': 'max'},
                 input=json.dumps(dict(context, continuation=continuation, incumbent_u=r.u.tolist()))))
    def consume(proposals, info, searches, baseline):
        whole.result.objective *= 1-drops.get(info['call_id'], 0.)
        info['feedback'] = {}
    monkeypatch.setattr(whole, 'consume_upper', consume)

    class Worker:
        def __init__(self, d, o, body, iteration, seconds, *, session_id=None, isolated=False):
            self.call = dict(model=o.llm_model, command=o.llm_codex_command, directory=o.llm_agent_directory,
                             timeout=o.llm_timeout, iteration=iteration, seconds=seconds, session_id=session_id,
                             isolated=isolated, prompt=json.loads(body['input']))
            self.started, self.receipt, self.closed = perf_counter(), {}, False
            workers.append(self)
        def poll(self):
            return replies.pop(self.call['iteration'], None)
        def close(self, info=None):
            self.closed = True
    monkeypatch.setattr(transport, 'AdvisorRequest', Worker)
    return NS(whole=whole, advice=whole.advice, o=options, drops=drops, replies=replies, workers=workers)


def ok(status='OK'):
    return [], dict(status=status, message='', background_seconds=1., proposals=[])


# 常规 #1 降幅 0.01%：同一步里常规 #2 照常发出，升级层 #3 以隔离、不续接的完整提示并行发出。
def start_parallel(lanes):
    lanes.advice.begin(); lanes.advice.advance([])
    lanes.drops['llm_upper:1'] = .0001; lanes.replies[1] = ok()
    lanes.advice.advance([])
    return lanes.workers


def test_regular_stall_starts_isolated_escalation_in_parallel(lanes):
    regular1, regular2, escalation = start_parallel(lanes)
    advice = lanes.advice
    assert advice.regular.worker is regular2 and advice.escalation.worker is escalation
    assert regular2.call['iteration'] == 2 and escalation.call['iteration'] == 3
    assert escalation.call == dict(escalation.call, model='gpt-6-astra', command='codex',
        directory=lanes.o.llm_escalation_directory, session_id=None, isolated=True)
    assert escalation.call['seconds'] == escalation.call['timeout'] == 1800
    assert escalation.call['prompt']['continuation'] is False
    assert 'advice_routing' not in escalation.call['prompt'] and not regular2.call['isolated']
    row = lanes.whole.result.llm_history[-1]
    assert row['tier'] == 'escalation' and row['triggered_by'] == 'llm_upper:1'
    assert row['deadline_seconds'] == 1800 and row['transport'] == 'codex_exec'
    assert lanes.whole.result.llm_history[0]['tier'] == 'regular'
    assert advice.state['escalation'] == dict(advice.state['escalation'], armed=False, armed_by=None, calls=1)


def test_batch_end_keeps_escalation_and_final_close_cancels_it(lanes):
    start_parallel(lanes)
    escalation = lanes.workers[2]
    lanes.drops['llm_upper:2'] = .01; lanes.replies[2] = ok()
    lanes.advice.advance([])
    assert lanes.advice.state['batch'] is None and not escalation.closed
    assert lanes.whole.diagnostics['llm_batches'][-1]['status'] == 'RETURN_TO_SA'
    lanes.advice.close()
    row = lanes.whole.result.llm_history[2]
    assert escalation.closed and lanes.advice.escalation.worker is None
    assert row['status'] == 'CANCELLED' and row['message'] == 'SearchFinished'


def test_escalation_reply_is_polled_without_a_regular_batch(lanes):
    start_parallel(lanes)
    lanes.drops['llm_upper:2'] = .01; lanes.replies[2] = ok()
    lanes.advice.advance([])
    lanes.drops['llm_escalation:3'] = .01; lanes.replies[3] = ok()
    lanes.advice.advance([])
    row = lanes.whole.result.llm_history[2]
    assert row['status'] == 'OK' and row['improved'] and lanes.advice.escalation.worker is None


@pytest.mark.parametrize('improves', [True, False])
def test_escalation_improvement_clears_arm_otherwise_restarts(lanes, improves):
    start_parallel(lanes)
    lanes.drops['llm_upper:2'] = .0001; lanes.replies[2] = ok()
    lanes.drops['llm_escalation:3'] = .01 if improves else 0.; lanes.replies[3] = ok()
    lanes.advice.advance([])
    state = lanes.advice.state['escalation']
    calls = [w.call['iteration'] for w in lanes.workers]
    if improves:
        assert calls == [1, 2, 3, 4] and not state['armed'] and state['last_improved_seconds'] is not None
        # 常规 #4 早于 astra 改进被消费时发出：它的回复不再武装升级层。
        lanes.drops['llm_upper:4'] = .0001; lanes.replies[4] = ok()
        lanes.advice.advance([])
        assert not state['armed'] and lanes.advice.escalation.worker is None
    else:
        assert calls == [1, 2, 3, 4, 5] and lanes.workers[-1].call['isolated']
        assert lanes.whole.result.llm_history[-1]['triggered_by'] == 'llm_upper:2'


@pytest.mark.parametrize('status,drop,started,armed', [
    ('OK', .004, 10., True), ('OK', .006, 10., False), ('FAILED', 0., 10., True),
    ('TIME_LIMIT', 0., 10., True), ('CANCELLED', 0., 10., False), ('OK', 0., 1., False)])
def test_arming_rule(lanes, status, drop, started, armed):
    lanes.advice.state['escalation']['last_improved_seconds'] = 5.
    lanes.advice.arm(dict(call_id='llm_upper:9', status=status, started_seconds=started,
                          upper_before_probes=100., upper_after_probes=100.*(1-drop)))
    assert lanes.advice.state['escalation']['armed'] is armed
    assert lanes.advice.state['escalation']['armed_by'] == ('llm_upper:9' if armed else None)


def test_reply_consumed_after_deadline_and_start_failure(lanes, monkeypatch):
    lanes.advice.begin(); lanes.advice.advance([])
    lanes.whole.deadline = perf_counter()-1
    lanes.replies[1] = ok()
    assert lanes.advice.poll([])
    assert lanes.whole.result.llm_history[0]['status'] == 'TIME_LIMIT'
    assert not lanes.advice.state['escalation']['armed']
    lanes.whole.deadline = perf_counter()+7200
    def broken(*args, **kwargs):
        raise OSError('spawn failed')
    monkeypatch.setattr(transport, 'AdvisorRequest', broken)
    lanes.advice.state.update(phase='sa', batch=None)
    lanes.advice.regular.worker = None
    lanes.advice.begin(); lanes.advice.advance_batch([])
    assert lanes.whole.result.llm_history[-1]['status'] == 'FAILED'
    assert lanes.advice.state['escalation']['armed_by'] == lanes.whole.result.llm_history[-1]['call_id']


def test_window_call_limit_and_convergence_block_escalation(lanes, monkeypatch):
    advice, state = lanes.advice, lanes.advice.state['escalation']
    state.update(armed=True, armed_by='llm_upper:1')
    lanes.whole.deadline = perf_counter()+750
    advice.advance_escalation([])
    assert not lanes.workers and state['armed']
    lanes.whole.deadline = perf_counter()+7200
    lanes.o.llm_max_calls = lanes.whole.result.llm_calls
    advice.advance_escalation([])
    assert not lanes.workers
    lanes.o.llm_max_calls = None
    monkeypatch.setattr(lanes.whole.result, 'reference_target_reached', lambda tolerance: True)
    advice.advance_escalation([])
    assert not lanes.workers
    monkeypatch.setattr(lanes.whole.result, 'reference_target_reached', lambda tolerance: False)
    lanes.whole.deadline = perf_counter()+1000
    advice.advance_escalation([])
    worker, = lanes.workers
    assert worker.call['seconds'] == pytest.approx(940, abs=1) and worker.call['timeout'] == worker.call['seconds']
    assert lanes.whole.result.llm_history[-1]['deadline_seconds'] == worker.call['seconds']


def test_two_escalation_failures_disable_only_escalation(lanes, monkeypatch):
    advice, state = lanes.advice, lanes.advice.state['escalation']
    for call in (1, 2):
        state.update(armed=True, armed_by='llm_upper:0')
        advice.advance_escalation([])
        lanes.replies[call] = ok('FAILED')
        advice.advance_escalation([])
    assert state['disabled'] and lanes.whole.diagnostics['llm_escalation_disabled']['reason']
    assert advice.state['disabled'] is None and advice.available()
    state.update(armed=True)
    advice.advance_escalation([])
    assert len(lanes.workers) == 2
    def broken(*args, **kwargs):
        raise ValueError('Unsupported Codex config value at profiles: list')
    monkeypatch.setattr(transport, 'AdvisorRequest', broken)
    state.update(disabled=None, consecutive_failures=0, armed=True)
    advice.advance_escalation([])
    row = lanes.whole.result.llm_history[-1]
    assert row['status'] == 'FAILED' and 'Unsupported Codex config' in row['message']
    assert state['consecutive_failures'] == 1 and advice.escalation.worker is None
    assert lanes.whole.result.llm_calls == 3


def test_escalation_options_are_validated(tmp_path):
    base = dict(llm_enabled=True, llm_model='deepseek', llm_url='http://relay', llm_api_key='key')
    missing = tmp_path/'none.json'
    with pytest.raises(ValueError, match='escalation'):
        load_options(dict(base, llm_escalation_model='gpt-6-astra'), missing)
    escalation = dict(base, llm_escalation_model='gpt-6-astra', llm_escalation_command='codex',
                      llm_escalation_directory=str(tmp_path))
    for name, value in (('llm_escalation_gain', -.1), ('llm_escalation_timeout', 0), ('llm_escalation_min_seconds', True)):
        with pytest.raises(ValueError, match='Invalid numeric option'):
            load_options(dict(escalation, **{name: value}), missing)
    with pytest.raises(ValueError, match='must be a string'):
        load_options(dict(base, llm_escalation_model=1), missing)
    o = load_options(dict(escalation, llm_escalation_gain=0, llm_escalation_min_seconds=0), missing)
    assert o.llm_escalation_gain == 0 and o.llm_escalation_min_seconds == 0


USER_CONFIG = r'''model_provider = "custom"
model = "gpt-6-astra"
notify = ["notify.exe"]
[model_providers.custom]
name = "OpenAI"
wire_api = "responses"
requires_openai_auth = true
[features]
memories = true
[memories]
use_memories = true
[plugins."ponytail@ponytail"]
enabled = true
[projects.'C:\Users\x\repo']
trust_level = "trusted"
[windows]
sandbox = "elevated"
path = "C:\\Tools\\bin"
[tui.model_availability_nux]
"gpt-6.1-sol" = 1
[agents]
'''


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    source = tmp_path/'user-codex'
    for name in ('memories', 'sessions'):
        (source/name).mkdir(parents=True)
    (source/'config.toml').write_text(USER_CONFIG, encoding='utf-8')
    (source/'auth.json').write_text('{"OPENAI_API_KEY": "secret"}', encoding='utf-8')
    (source/'AGENTS.md').write_text('private instructions', encoding='utf-8')
    (tmp_path/'tmp').mkdir()
    monkeypatch.setenv('CODEX_HOME', str(source))
    monkeypatch.setattr(transport.tempfile, 'tempdir', str(tmp_path/'tmp'))
    return source


def test_isolated_home_keeps_login_and_provider_only(codex_home, tmp_path):
    import tomli
    home = transport.isolated_home()
    assert home.parent == tmp_path/'tmp' and home.name.startswith('suc-codex-')
    assert sorted(p.name for p in home.iterdir()) == ['auth.json', 'config.toml']
    config = tomli.loads((home/'config.toml').read_text(encoding='utf-8'))
    assert not {'features', 'memories', 'plugins', 'projects', 'notify'} & config.keys()
    assert config['model_providers']['custom']['wire_api'] == 'responses'
    assert config['windows']['path'] == 'C:\\Tools\\bin' and config['agents'] == {}
    assert config['tui']['model_availability_nux']['gpt-6.1-sol'] == 1


@pytest.mark.parametrize('extra', ['[[profiles]]\nname = "a"\n', 'x = [{a = 1}]\n', 'when = 1979-05-27T07:32:00Z\n'])
def test_unsupported_codex_config_fails_without_leftovers(codex_home, tmp_path, extra):
    (codex_home/'config.toml').write_text(extra+USER_CONFIG if extra.startswith(('x', 'when')) else USER_CONFIG+extra,
                                          encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported Codex config'):
        transport.isolated_home()
    assert not list((tmp_path/'tmp').iterdir())


def test_prepare_with_isolated_home_is_ephemeral(codex_home, tmp_path, monkeypatch):
    monkeypatch.setattr(transport.shutil, 'which', lambda command: sys.executable)
    (codex_home/'config.toml').write_text('model = "x"\n', encoding='utf-8')
    home = tmp_path/'home'; home.mkdir()
    (home/'config.toml').write_text('model_provider = "custom"\n', encoding='utf-8')
    body = dict(instructions='Only return candidates', input='{"upper":100}', model='gpt-6-astra',
                reasoning={'effort': 'max'})
    directory = tmp_path/'request'; directory.mkdir()
    args, environment, _ = transport._prepare_codex('codex', body, NS(n_units=1, horizon=1), directory, None, home)
    assert '--ephemeral' in args and 'resume' not in args
    assert environment['CODEX_HOME'] == str(home)
    assert 'model_providers.custom.supports_websockets=false' in args and 'model_provider="suc_https"' not in args


class FakeProcess:
    def __init__(self, target, args, daemon):
        self.args, self.pid = args, 4242
    def start(self):
        pass
    def is_alive(self):
        return False
    def join(self, timeout=None):
        pass


class FakeConnection:
    def poll(self):
        return False
    def close(self):
        pass


def isolated_request(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(transport.mp, 'get_context', lambda method: NS(
        Pipe=lambda duplex: (FakeConnection(), FakeConnection()), Process=FakeProcess))
    o = NS(llm_agent_directory=str(tmp_path/'results'/'llm_requests'), llm_codex_command='codex',
           llm_url='', llm_api_key='')
    body = dict(model='gpt-6-astra', reasoning={'effort': 'max'}, instructions='', input='{"incumbent_u":[[0]]}')
    return transport.AdvisorRequest(NS(n_units=1, horizon=1), o, body, 3, 100., isolated=True, **kwargs)


def test_isolated_request_removes_home_on_close(codex_home, tmp_path, monkeypatch):
    request = isolated_request(tmp_path, monkeypatch)
    home = request.process.args[-1]
    assert home == request.home and (home/'auth.json').exists()
    request.close()
    assert not home.exists()
    receipt = json.loads((request.codex_directory/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['codex_home_cleanup'] == 'REMOVED' and receipt['isolated_codex_home'] is True
    assert receipt['requested_session_id'] is None
    assert not list((tmp_path/'results').rglob('auth.json'))


def test_cleanup_failure_is_recorded_not_raised(codex_home, tmp_path, monkeypatch):
    request = isolated_request(tmp_path, monkeypatch)
    monkeypatch.setattr(transport.shutil, 'rmtree', lambda *a, **k: None)
    monkeypatch.setattr(transport, 'sleep', lambda seconds: None)
    request.close()
    assert request.receipt['codex_home_cleanup'].startswith('FAILED')


def test_isolated_request_rejects_session_resume(codex_home, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='fresh Codex CLI task'):
        isolated_request(tmp_path, monkeypatch, session_id=SESSION)
    assert not list((tmp_path/'tmp').iterdir())
