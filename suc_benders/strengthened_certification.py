"""A cold, fixed numerical proof in a separate process; only bounds cross back."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter, sleep, time

import gurobipy as gp
import numpy as np

from .benders import master
from .certification import Certification
from .model import fingerprint, model, partition
from . import strengthened_master as strong
from .subagent_advisor import save


def matrix_hash(m):
    digest = hashlib.sha256(b'strengthened-matrix-v1')
    a = m.A.copy(); a.sort_indices()
    for values in (a.shape, a.indptr, a.indices, a.data, m.rhs, m.eq, m.obj, m.q,
                   m.lb, m.ub, m.binary, [getattr(m, 'constant', 0.)]):
        values = np.asarray(values, dtype='<f8')
        digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def save_matrix(path, m):
    np.savez_compressed(path, data=m.A.data, indices=m.A.indices, indptr=m.A.indptr,
        shape=m.A.shape, rhs=m.rhs, eq=m.eq, obj=m.obj, q=m.q, lb=m.lb, ub=m.ub,
        binary=m.binary, constant=getattr(m, 'constant', 0.))


def _proof(sender, stop, d, scenarios, deadline, started, directory):
    folder = Path(directory) if directory else None
    trace = None
    try:
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            trace = (folder/'trace.jsonl').open('w', encoding='utf-8', buffering=1)
        physical = model(d, scenarios)
        identity = fingerprint(physical)
        proof_hash = None

        def report(event, **values):
            row = dict(event=event, elapsed_seconds=perf_counter()-started,
                       completed_epoch_seconds=time(), physical_fingerprint=identity,
                       model_hash=proof_hash, **values)
            if trace:
                trace.write(json.dumps(row, allow_nan=False)+'\n')
            sender.send(row)

        last = [0.]
        def callback(g, where):
            if stop.is_set() or perf_counter() >= deadline:
                g.terminate()
            elif where == gp.GRB.Callback.MIP and perf_counter()-last[0] >= 5.:
                lower = float(g.cbGet(gp.GRB.Callback.MIP_OBJBND))+proof_constant-1e-5
                if np.isfinite(lower) and abs(lower) < 1e100:
                    report('bound', lower=lower, origin='MIP_OBJBND')
                last[0] = perf_counter()

        full = partition(physical, d.horizon)[0][0]
        root, _ = master(full, [])
        root = strong.add_temporal(root, physical)
        base = strong.build(physical, root, len(root.obj)-1)
        proof_constant = base.constant
        report('started', base_hash=matrix_hash(base), process_seed=7)
        if folder:
            save_matrix(folder/'base.npz', base)
        seconds = min(180., deadline-perf_counter()-.5)
        if seconds <= 0 or stop.is_set():
            report('stopped', status='INTERRUPTED')
            return
        refined = strong.separate(physical, base, seconds, .01, callback=callback)
        proof_hash = matrix_hash(refined.model)
        preparation = dict(physical_fingerprint=identity, base_hash=matrix_hash(base),
            model_hash=proof_hash, status=refined.status, separation_seconds=refined.elapsed_seconds,
            perspective_gap=float(refined.gap) if np.isfinite(refined.gap) else None,
            cuts=refined.cuts, integer_call_limit=1, process_seed=7, warm_start=None,
            inherited_evidence=False, upper_feedback=False)
        if folder:
            save_matrix(folder/'model.npz', refined.model)
            save(folder/'preparation.json', preparation)
        report('prepared', separation_status=refined.status, cuts=len(refined.cuts))
        if np.isfinite(refined.lower):
            report('bound', lower=float(refined.lower), origin='continuous_separation')
        seconds = deadline-perf_counter()-.5
        if seconds <= 0 or stop.is_set():
            report('stopped', status='INTERRUPTED')
            return
        result = strong.certify(refined.model, seconds, callback=callback,
            log_file=str(folder/'gurobi.log') if folder else None,
            model_file=folder/'model.mps' if folder else None)
        values = dict(status=result.status, lower=result.lower if np.isfinite(result.lower) else None,
                      solver_wall_seconds=result.elapsed_seconds, optimizer_calls=result.optimizer_calls)
        if folder:
            save(folder/'solve.json', dict(values, model_hash=proof_hash))
        report('finished', **values)
    except Exception as error:
        try:
            sender.send(dict(event='failed', message=type(error).__name__,
                             elapsed_seconds=perf_counter()-started))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if trace:
            trace.close()
        sender.close()


class StrengthenedCertification(Certification):
    """Reuse incumbent checks, but never pass numerical candidates to WholeDay."""
    def __init__(self, whole):
        super().__init__(whole)
        self.process = self.connection = self.stop = None
        self.finished = False
        whole.diagnostics['strengthened_proof'] = dict(status='NOT_STARTED', bounds=[], upper_feedback=False)

    def start(self):
        if self.process is not None:
            return
        context = mp.get_context('spawn')
        self.connection, sender = context.Pipe(duplex=False)
        self.stop = context.Event()
        directory = getattr(self.whole.progress, 'proof_directory', None)
        self.process = context.Process(target=_proof, args=(sender, self.stop, self.whole.d,
            self.whole.scenarios, self.whole.deadline, self.whole.started, directory), daemon=True)
        try:
            self.process.start()
        finally:
            sender.close()

    def observe(self, x, evaluation, source):
        if evaluation.feasible:
            evaluation.hourly_recourse = self.hourly.recourse(evaluation.point)

    def inherit(self, evidence):
        raise ValueError('Strengthened proof requires its own cold model')

    def restore_cuts(self, diagnostic):
        if diagnostic.get('certificate_cuts') or diagnostic.get('hourly_certificate_cuts'):
            raise ValueError('Strengthened proof cannot import Benders cuts')
        if 'strengthened_proof' not in diagnostic:
            raise ValueError('Recovered state is not a strengthened proof run')
        self.whole.diagnostics['strengthened_proof']['prior_proof'] = diagnostic['strengthened_proof']

    def poll(self):
        if self.connection is None or self.finished:
            return
        state = self.whole.diagnostics['strengthened_proof']
        while True:
            try:
                if not self.connection.poll():
                    break
                row = self.connection.recv()
            except (EOFError, OSError):
                break
            if row['event'] == 'failed':
                state.update(status='FAILED', message=row['message'])
                self.finished = True
                raise RuntimeError('Strengthened proof worker failed: '+row['message'])
            if row.get('physical_fingerprint') != self.whole.diagnostics['model_fingerprint']:
                raise ValueError('Strengthened proof model identity mismatch')
            state['status'] = row['event']
            if row.get('model_hash'):
                state['model_hash'] = row['model_hash']
            usable = (row['elapsed_seconds'] <= self.whole.deadline-self.whole.started
                      and self.whole.remaining() > 0)
            lower = row.get('lower')
            if lower is not None and usable:
                self.update_bound(lower)
                state['bounds'].append(row)
                self.result.certification_history.append(dict(iteration=len(self.result.certification_history)+1,
                    status=row.get('status', row['event']), lower=self.result.lower_bound,
                    upper=self.result.objective, elapsed_seconds=perf_counter()-self.whole.started,
                    produced_seconds=row['elapsed_seconds'], master_bound=lower, bound_used=True,
                    source='strengthened_numeric', model_hash=row['model_hash']))
                self.whole.emit('certified_bound')
            if row['event'] in ('finished', 'stopped'):
                self.finished = True
                state['solver_status'] = row['status']
                if row['status'] not in ('OPTIMAL', 'TIME_LIMIT', 'INTERRUPTED', 'USER_OBJ_LIMIT'):
                    raise RuntimeError('Invalid strengthened proof status: '+row['status'])
        if not self.finished and not self.process.is_alive():
            raise RuntimeError('Strengthened proof worker exited without a terminal receipt')

    def step(self):
        self.start()
        self.poll()
        sleep(min(.05, max(0., self.whole.remaining())))

    def close(self):
        if self.process is None:
            return
        self.stop.set()
        self.process.join(timeout=4.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
            self.whole.diagnostics['strengthened_proof']['shutdown'] = 'FORCED_STOP'
        else:
            self.poll()
        self.connection.close()
        if not self.finished:
            self.whole.diagnostics['strengthened_proof']['shutdown'] = 'NO_TERMINAL_RECEIPT'
