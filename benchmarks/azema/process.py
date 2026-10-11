"""Single-process-tree watchdog. Only processes launched by this benchmark are stopped."""
import os
import json
from pathlib import Path
import subprocess
from time import monotonic, sleep

import psutil

from . import ROOT
from .artifacts import append, memory_gate, save


def single_thread_environment(extra=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='7',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    env.update(extra or {})
    return env


def stop_tree(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs([*children, parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def run_process(command, directory, seconds, *, env=None, numerical=True, estimated_peak_bytes=0,
                enforce_memory_gate=True):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    gate = memory_gate(estimated_peak_bytes)
    gate['enforced'] = enforce_memory_gate
    save(directory/'resource_gate.json', gate)
    if numerical and enforce_memory_gate and not gate['passed']:
        state = dict(status='BLOCKED_MEMORY', numerical_runs_started=0, llm_calls_started=0, **gate)
        save(directory/'state.json', state)
        return state
    if seconds <= 0:
        raise ValueError('No process budget remaining')
    start = monotonic(); peak = 0; reason = None
    with (directory/'process.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=single_thread_environment(env), stdout=log,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(directory/'state.json', dict(status='RUNNING', pid=process.pid, command=command,
            budget_seconds=seconds, numerical_runs_started=int(numerical)))
        try:
            parent = psutil.Process(process.pid)
            while process.poll() is None:
                elapsed = monotonic()-start
                try:
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                peak = max(peak, rss)
                available = psutil.virtual_memory().available
                append(directory/'monitor.jsonl', dict(elapsed_seconds=elapsed, rss_bytes=rss, available_bytes=available))
                if elapsed >= seconds:
                    reason = 'TIME_LIMIT'
                elif numerical and enforce_memory_gate and available < 512*1024**2:
                    reason = 'RESOURCE_EXHAUSTED'
                if reason:
                    stop_tree(process)
                    break
                sleep(min(.5, max(.001, seconds-elapsed)))
        except BaseException:
            stop_tree(process)
            raise
    state = dict(status=reason or ('FINISHED' if process.returncode == 0 else 'FAILED'),
        returncode=process.returncode, wall_seconds=monotonic()-start, peak_rss_bytes=peak,
        numerical_runs_started=int(numerical), budget_seconds=seconds)
    worker_state = json.loads((directory/'state.json').read_text(encoding='utf-8'))
    if worker_state.get('status') == 'BLOCKED_MEMORY':
        state.update(status='BLOCKED_MEMORY', numerical_runs_started=0, llm_calls_started=0,
                     worker_gate=json.loads((directory/'resource_gate.json').read_text(encoding='utf-8')))
    save(directory/'state.json', state)
    return state
