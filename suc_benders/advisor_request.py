"""One cancellable upper-advice request over streaming HTTP or the Codex CLI."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import tomli
from time import perf_counter, sleep, time
from uuid import UUID, uuid4
from types import SimpleNamespace as NS

from . import llm
from .data import save_json


class SessionUnavailable(RuntimeError):
    """A saved conversation cannot be continued without changing identity."""


def session_file(session_id):
    try:
        UUID(session_id)
    except (ValueError, TypeError, AttributeError):
        raise SessionUnavailable('Invalid Codex session ID') from None
    home = Path(os.environ.get('CODEX_HOME', Path.home()/'.codex'))
    for root in (home/'sessions', home/'archived_sessions'):
        for path in root.glob(f'**/*{session_id}.jsonl'):
            return path
    raise SessionUnavailable('Codex session file is missing: '+session_id)

# 单个在途请求的生命周期对象，提供非阻塞 poll 和终止/关闭操作。
class AdvisorRequest:
    # 设置了 Codex 命令时走 CLI，否则走 HTTP；两者都在独立工作进程中执行。
    # 传给子进程的数据只包含请求体和矩阵形状，避免传递求解器对象。
    # isolated=True 时由父进程建立临时 CODEX_HOME（见 isolated_home），关闭请求时删除；只用于不续接的新任务。
    def __init__(self, d, o, body, iteration, seconds, *, session_id=None, isolated=False):
        self.codex_directory = None
        self.home = None
        self.closed = False
        if isolated and (not o.llm_codex_command or session_id is not None):
            raise ValueError('Isolated requests require a fresh Codex CLI task')
        self.started = perf_counter()
        self.deadline = self.started + seconds
        context = mp.get_context("spawn")
        self.connection, sender = context.Pipe(duplex=False)
        # base_u 取自请求体中的 incumbent_u，保证增量候选按模型看到的方案还原。
        shape = NS(n_units=d.n_units, horizon=d.horizon, base_u=llm.prompt_base(body))
        self.receipt = dict(request=body, request_sha256=hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        if o.llm_codex_command:
            self.codex_directory = Path(o.llm_agent_directory).resolve()/f'{iteration:02d}-{uuid4().hex}'
            self.codex_directory.mkdir(parents=True)
            self.receipt.update(transport='codex_exec', request_directory=str(self.codex_directory),
                                requested_model=body['model'], reasoning_effort=body['reasoning']['effort'],
                                requested_session_id=session_id, isolated_codex_home=isolated)
            save_json(self.codex_directory/'request.json', dict(**self.receipt,
                shape=[d.n_units, d.horizon], created_epoch_seconds=time(), timeout_seconds=seconds))
            target, args = _request_codex, (sender, o.llm_codex_command, body, shape, iteration,
                                            self.deadline, self.codex_directory, session_id)
        else:
            target, args = _request, (sender, o.llm_url, o.llm_api_key, body, shape, iteration, seconds)
        try:
            if isolated:
                self.home = isolated_home()
                args += (self.home,)
            self.process = context.Process(target=target, args=args, daemon=True)
            self.process.start()
        except Exception as error:
            self.connection.close()
            if self.home is not None:
                self.receipt['codex_home_cleanup'] = _remove_home(self.home)
            self.save_receipt(dict(status='FAILED', message=f'{type(error).__name__}: {error}',
                accepted=0, proposals=[], seconds=perf_counter()-self.started, **self.receipt))
            raise
        finally:
            # 父进程释放自己的发送端副本，保证子进程结束后读端能正确观察到 EOF。
            sender.close()

    # 尚未完成时返回 None；完成、超时或进程退出时返回 (候选, 回执) 并回收资源。
    # 完成时刻晚于请求截止时间的回复必须丢弃，即使当前已经可以读到结果。
    def poll(self):
        if self.closed:
            return [], dict(status='CANCELLED', message='RequestClosed', accepted=0, proposals=[],
                            seconds=0., background_seconds=0., **self.receipt)
        now = perf_counter()
        if self.connection.poll():
            try:
                candidates, info = self.connection.recv()
            except EOFError:
                candidates, info = [], dict(status="TIME_LIMIT" if now >= self.deadline else "FAILED",
                                           message="WorkerExited", accepted=0, proposals=[], finished_at=now)
            if info["finished_at"] > self.deadline:
                candidates = []
                info.update(status="TIME_LIMIT", accepted=0)
            info["background_seconds"] = max(0., info["finished_at"]-self.started)
            if self.codex_directory is not None:
                info['seconds'] = info['background_seconds']
            info.update(self.receipt)
            self.close(info)
            return candidates, info
        if now >= self.deadline or not self.process.is_alive():
            info = dict(status="TIME_LIMIT" if now >= self.deadline else "FAILED",
                        message="RequestDeadline" if now >= self.deadline else "WorkerExited",
                        seconds=now-self.started, background_seconds=now-self.started,
                        accepted=0, proposals=[], **self.receipt)
            self.close(info)
            return [], info
        return None

    def save_receipt(self, info):
        if self.codex_directory is not None:
            info.update(_codex_event_info(self.codex_directory))
            info.setdefault('exit_code', None)
            save_json(self.codex_directory/'receipt.json', info)

    # CLI 的工作进程连同其 Node/native 子进程一并回收，不能只终止 Python 父进程。
    def close(self, info=None):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            if self.codex_directory is not None:
                _kill_tree(self.process.pid)
            self.process.terminate()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)
        self.connection.close()
        # 进程树结束后再删除隔离目录（其中有登录信息副本）；删除结果写入回执，失败不抛出。
        if self.home is not None:
            self.receipt['codex_home_cleanup'] = _remove_home(self.home)
            if info is not None:
                info['codex_home_cleanup'] = self.receipt['codex_home_cleanup']
        self.save_receipt(info if info is not None else dict(status='CANCELLED', message='RequestClosed',
            accepted=0, proposals=[], exit_code=None, seconds=perf_counter()-self.started, **self.receipt))


def _kill_tree(pid):
    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   creationflags=subprocess.CREATE_NO_WINDOW, check=False)


# 隔离的升级层调用从临时 CODEX_HOME 去掉这些配置段：记忆、插件、hooks、MCP、桌面与项目设置。
# 全局 AGENTS.md、memories 和会话记录不复制，只保留登录信息与模型提供方等其余设置。
ISOLATED_DROP = ('features', 'memories', 'plugins', 'hooks', 'mcp_servers', 'marketplaces', 'desktop',
                 'projects', 'notify')


def _toml_value(value, path):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return '['+', '.join(_toml_value(v, path) for v in value)+']'
    raise ValueError(f'Unsupported Codex config value at {".".join(path)}: {type(value).__name__}')


def _toml_table(table, path, lines):
    if path:
        lines.append('['+'.'.join(json.dumps(k, ensure_ascii=False) for k in path)+']')
    lines += [json.dumps(k, ensure_ascii=False)+' = '+_toml_value(v, (*path, k))
              for k, v in table.items() if not isinstance(v, dict)]
    for k, v in table.items():
        if isinstance(v, dict):
            _toml_table(v, (*path, k), lines)


# 建立临时 CODEX_HOME：过滤后的 config.toml 与 auth.json 副本，返回路径，由调用方负责删除。
# 写出前回读校验，数组表、日期等无法原样写出的结构直接报错，不生成残缺配置。
def isolated_home():
    source = Path(os.environ.get('CODEX_HOME', Path.home()/'.codex'))
    path = source/'config.toml'
    config = tomli.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for key in ISOLATED_DROP:
        config.pop(key, None)
    lines = []
    _toml_table(config, (), lines)
    text = '\n'.join(lines)+'\n'
    if tomli.loads(text) != config:
        raise ValueError('Filtered Codex config does not round-trip')
    home = Path(tempfile.mkdtemp(prefix='suc-codex-'))
    try:
        (home/'config.toml').write_text(text, encoding='utf-8')
        if (source/'auth.json').exists():
            shutil.copy2(source/'auth.json', home/'auth.json')
    except BaseException:
        shutil.rmtree(home, ignore_errors=True)
        raise
    return home


# Windows 上 CLI 退出后文件句柄可能稍晚释放，重试删除；返回 REMOVED 或残留路径，不抛异常。
def _remove_home(home):
    for _ in range(10):
        shutil.rmtree(home, ignore_errors=True)
        if not Path(home).exists():
            return 'REMOVED'
        sleep(.2)
    return f'FAILED: {home}'


def _codex_https_args(config_path):
    config = tomli.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    provider = config.get('model_provider', 'openai')
    # 内置 openai 不允许原地覆盖，使用仅限本次调用的 HTTPS 提供方。
    if provider == 'openai':
        return ['-c', 'model_provider="suc_https"', '-c',
                'model_providers.suc_https={name="OpenAI HTTPS",wire_api="responses",'
                'requires_openai_auth=true,supports_websockets=false}']
    return ['-c', f'model_providers.{provider}.supports_websockets=false']


def _codex_event_info(directory):
    path = directory/'events.jsonl'
    info = {}
    if not path.exists():
        return info
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # 进程被回收时，最后一行可能只写了一部分。
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'thread.started':
            info['thread_id'] = event['thread_id']
        elif event.get('type') == 'turn.completed':
            info['usage'] = event.get('usage')
        elif event.get('type') in ('error', 'turn.failed'):
            message = event.get('message', '') or event.get('error', {}).get('message', '')
            message = re.sub(r'sk-[A-Za-z0-9_*\-]+', '[REDACTED_KEY]', message)
            info['last_cli_error'] = re.sub(r'Bearer\s+[^\s,\"]+', 'Bearer [REDACTED]', message)
    return info


def _prepare_codex(command, body, shape, directory, session_id=None, home=None):
    """Prepare the same isolated CLI request for production and timing experiments."""
    # home 为隔离的临时 CODEX_HOME：不落盘会话（--ephemeral），提供方设置读自其中的过滤配置。
    # 续接的增量提示没有 instructions，不能留下空段落。
    prompt = ((body['instructions']+'\n\n' if body['instructions'] else '')+body['input']
                  +'\n\nReturn only the requested JSON. Do not invoke tools, edit files, or spawn agents.\n')
    if session_id is not None:
        session_file(session_id)
        prompt = ('Continue this conversation. The latest snapshot and measured feedback below supersede '
                  'older states. Previous proposals have already been evaluated; propose new improvements.\n\n'+prompt)
    (directory/'prompt.txt').write_bytes(prompt.encode('utf-8'))
    prompt_sha256 = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    save_json(directory/'schema.json', llm.output_schema(shape.n_units, shape.horizon))
    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError('Codex CLI not found; install it or set llm_codex_command')
    args = [executable, 'exec', '--model', body['model'],
                '-c', 'model_reasoning_effort="'+body['reasoning']['effort']+'"',
                '-c', 'approval_policy="never"', '-c', 'agents.enabled=false',
                '--sandbox', 'read-only', '--cd', str(directory)]
    if session_id is not None:
        args += ['resume', session_id]
    args += ['--skip-git-repo-check', '--json', '--output-schema', str(directory/'schema.json'),
             '--output-last-message', str(directory/'reply.json'), '-']
    config_path = Path(home if home is not None else os.environ.get('CODEX_HOME', Path.home()/'.codex'))/'config.toml'
    args[2:2] = _codex_https_args(config_path)
    if home is not None:
        args.insert(2, '--ephemeral')
    save_json(directory/'command.json', dict(argv=args))
    environment = os.environ.copy()
    # codex exec 会优先使用此覆盖值；复用 CLI 已有登录时不能继承父代理的临时密钥。
    environment.pop('CODEX_API_KEY', None)
    if home is not None:
        environment['CODEX_HOME'] = str(home)
    return args, environment, prompt_sha256


def _request_codex(connection, command, body, shape, iteration, deadline, directory, session_id=None, home=None):
    # Ctrl+C 由主进程统一回收整棵进程树，工作进程不能先于 CLI 退出。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    started = perf_counter()
    candidates, process = [], None
    info = dict(status='FAILED', message='', accepted=0, format_valid=0, proposals=[], exit_code=None)
    try:
        args, environment, info['prompt_sha256'] = _prepare_codex(command, body, shape, directory, session_id, home)
        with (directory/'prompt.txt').open('rb') as source, (directory/'events.jsonl').open('wb') as events, \
                (directory/'stderr.log').open('wb') as errors:
            if perf_counter() >= deadline:
                raise subprocess.TimeoutExpired(args, 0.)
            process = subprocess.Popen(args, stdin=source, stdout=events, stderr=errors,
                env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
            info['cli_pid'] = process.pid
            info['exit_code'] = process.wait(timeout=max(0., deadline-perf_counter()))
        info.update(_codex_event_info(directory))
        if session_id is not None and info.get('thread_id') not in (None, session_id):
            raise SessionUnavailable('Codex returned a mismatched session ID')
        if info['exit_code'] != 0:
            raise RuntimeError('Codex CLI exited with code '+str(info['exit_code']))
        if not info.get('thread_id') or (session_id is not None and info['thread_id'] != session_id):
            raise SessionUnavailable('Codex returned a missing or mismatched session ID')
        reply = (directory/'reply.json').read_bytes()
        info['reply_sha256'] = hashlib.sha256(reply).hexdigest()
        text = reply.decode('utf-8')
        raw = dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text=text)])])
        candidates, parsed = llm.parse_response(raw, shape, iteration, shape.base_u)
        info.update(parsed, response_output=[text])
    except subprocess.TimeoutExpired:
        info.update(status='TIME_LIMIT', message='CodexDeadline')
    except SessionUnavailable as error:
        info.update(status='SESSION_FAILED', message=str(error))
    except Exception as error:
        info.update(status='FAILED', message=f'{type(error).__name__}: {error}')
    finished = perf_counter()
    if finished > deadline:
        candidates = []
        info.update(status='TIME_LIMIT', accepted=0, message='CodexDeadline')
    info.update(seconds=finished-started, finished_at=finished)
    try:
        connection.send((candidates, info))
    finally:
        connection.close()
        # 保持父子关系，直到消费方收到超时回执并回收整个进程树。
        if process is not None and process.poll() is None:
            process.wait()


def _request(connection, url, key, body, shape, iteration, seconds):
    started = perf_counter()
    candidates = []
    info = dict(status="FAILED", message="", accepted=0, format_valid=0, proposals=[])
    try:
        raw = llm.post_response(url, key, body, seconds)
        candidates, info = llm.parse_response(raw, shape, iteration, shape.base_u)
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
