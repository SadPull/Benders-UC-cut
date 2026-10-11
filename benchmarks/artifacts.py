"""Durable evidence, resource gates and source snapshots for paper C."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import psutil

from suc_benders.model import fingerprint
from . import ROOT


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if hasattr(value, '__dict__'):
        return serializable(vars(value))
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(serializable(value), stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(serializable(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def memory_gate(estimated_peak_bytes=0):
    memory = psutil.virtual_memory()
    required = max(4*1024**3, int(estimated_peak_bytes)+2*1024**3)
    return dict(passed=memory.available >= required, available_bytes=memory.available,
                required_bytes=required, estimated_peak_bytes=estimated_peak_bytes)


def source_hashes():
    paths = [ROOT/'run_suc.py', ROOT/'run_rts24_paper_c.py', ROOT/'requirements.txt', ROOT/'pytest.ini']
    paths += list((ROOT/'suc_benders').glob('*.py'))
    paths += [p for p in (ROOT/'benchmarks').rglob('*') if p.suffix == '.py']
    paths += list((ROOT/'tests').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths) if p.exists()}


def snapshot(folder):
    hashes = source_hashes()
    for name in hashes:
        target = Path(folder)/'snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    save(Path(folder)/'source_sha256.json', hashes)
    return hashes


class Recorder:
    audit_cuts = True
    full_diagnostics = True
    streams = ('evaluations', 'proposals', 'sampling', 'certificate_cuts', 'hourly_certificate_cuts',
               'bound_probes', 'rounds', 'local_evaluations', 'certification_history')

    def __init__(self, directory, time_limit):
        self.directory, self.time_limit = Path(directory), time_limit
        self.directory.mkdir(parents=True, exist_ok=True)
        self.offsets = {name: 0 for name in self.streams}
        self.last, self.peak_rss = None, 0
        self.process = psutil.Process()
        self.sequence = 0

    def relaxation(self, m, value, elapsed):
        target = self.directory/'initial_prices.npz'
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, prices=value.prices)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        save(self.directory/'initial_bound.json', dict(lower_bound=value.lower,
            model_fingerprint=fingerprint(m), arrays_sha256=sha256(target), elapsed_seconds=elapsed))

    def __call__(self, whole, event):
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        elapsed = perf_counter()-whole.started
        if event == 'before_sampling':
            q = whole.diagnostics['pending_qubo']
            if not np.isfinite(q['max_coefficient']):
                raise RuntimeError('Nonfinite QUBO coefficient')
            if (not getattr(self, 'allow_low_memory', False)
                    and psutil.virtual_memory().available < q['dense_working_bytes']+512*1024**2):
                raise MemoryError('Dense Kaiwu matrix exceeds available memory margin')
        for name, offset in self.offsets.items():
            rows = (whole.result.certification_history if name == 'certification_history'
                    else whole.diagnostics.get(name, []))
            for row in rows[offset:]:
                append(self.directory/(name+'.jsonl'), row)
            self.offsets[name] = len(rows)
        r = whole.result
        if perf_counter()-whole.started <= self.time_limit:
            lower, upper = float(r.lower_bound), float(r.objective)
            if np.isfinite(lower) and np.isfinite(upper) and lower > upper+1e-5:
                raise RuntimeError('Recorded certified bound exceeds upper bound')
            row = dict(event=event, elapsed_seconds=elapsed, lower_bound=lower, objective=upper,
                       relative_gap=(upper-lower)/max(1., abs(upper)) if np.isfinite(upper) else None,
                       sa_calls=whole.diagnostics['sa_calls'], llm_calls=r.llm_calls, rss_bytes=rss)
            self.sequence += 1
            checkpoint = self.directory/'checkpoints'/f'{self.sequence:08d}.json'
            value = dict(**row, u=r.u, validation=r.validation, status=r.status,
                model_fingerprint=whole.diagnostics['model_fingerprint'], offsets=self.offsets.copy(),
                certificate=(r.certification_history[-1] if r.certification_history else None),
                timings=whole.timings.copy(), complete_rounds=whole.diagnostics['complete_rounds'],
                coordination_count=whole.diagnostics['coordination_count'])
            save(checkpoint, value)
            persisted = perf_counter()-whole.started
            # An immutable schedule must finish saving within budget to enter the scored trace.
            if persisted <= self.time_limit:
                row.update(elapsed_seconds=persisted, checkpoint=checkpoint.relative_to(self.directory).as_posix(),
                           checkpoint_sha256=sha256(checkpoint))
                append(self.directory/'trace.jsonl', row)
                self.last = dict(value, **{k: v for k, v in row.items()})
        if self.last is not None:
            save(self.directory/'checkpoint.json', dict(self.last, status=r.status, peak_rss_bytes=self.peak_rss))
        # Receipts/proposals can be updated after launch; this versioned snapshot is authoritative.
        save(self.directory/'llm_receipts.json', r.llm_history)
        if event == 'final' and self.full_diagnostics:
            save(self.directory/'diagnostics.json', whole.diagnostics)
            save(self.directory/'certification_history.json', r.certification_history)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8') as stream:
        rows = []
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed writer may leave one incomplete last line, never a committed checkpoint.
                if stream.read().strip():
                    raise
                break
        return rows


def scored_trace(directory, budget):
    directory = Path(directory)
    rows = []
    for row in read_jsonl(directory/'trace.jsonl'):
        if row['elapsed_seconds'] > budget:
            continue
        if row.get('checkpoint'):
            path = (directory/row['checkpoint']).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256(path) != row['checkpoint_sha256']:
                raise ValueError('Checkpoint path or hash mismatch')
        if row.get('objective') is not None and row.get('lower_bound') is not None:
            rows.append(row)
    return rows
