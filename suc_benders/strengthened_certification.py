"""A cold, fixed numerical proof in a separate process; only bounds cross back."""
import hashlib
import multiprocessing as mp
from time import perf_counter, sleep, time

import gurobipy as gp
import numpy as np

from .benders import master
from .model import fingerprint, model, partition
from . import strengthened_master as strong


def matrix_hash(m):
    digest = hashlib.sha256(b'strengthened-matrix-v1')
    a = m.A.copy(); a.sort_indices()
    for values in (a.shape, a.indptr, a.indices, a.data, m.rhs, m.eq, m.obj, m.q,
                   m.lb, m.ub, m.binary, [m.constant]):
        values = np.asarray(values, dtype='<f8')
        digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


# 证明进程主体：建立冷启动强化模型、做透视割分离，再以单次整数求解认证下界；只经管道回传数值事件。
def _proof(sender, stop, d, scenarios, deadline, started):
    try:
        physical = model(d, scenarios)
        identity = fingerprint(physical)
        proof_hash = None

        def report(event, **values):
            sender.send(dict(event=event, elapsed_seconds=perf_counter()-started,
                             completed_epoch_seconds=time(), physical_fingerprint=identity,
                             model_hash=proof_hash, **values))

        full = partition(physical, d.horizon)[0][0]
        root, _ = master(full, [])
        root = strong.add_temporal(root, physical)
        base = strong.build(physical, root, len(root.obj)-1)
        last_bound = 0.

        # 停止或到期时终止 Gurobi；整数求解中每 5 秒上报一次加回常数的 ObjBound。
        def callback(g, where):
            nonlocal last_bound
            if stop.is_set() or perf_counter() >= deadline:
                g.terminate()
            elif where == gp.GRB.Callback.MIP and perf_counter()-last_bound >= 5.:
                lower = float(g.cbGet(gp.GRB.Callback.MIP_OBJBND))+base.constant-1e-5
                if np.isfinite(lower) and abs(lower) < 1e100:
                    report('bound', lower=lower, origin='MIP_OBJBND')
                last_bound = perf_counter()

        report('started', base_hash=matrix_hash(base), process_seed=7)
        seconds = min(180., deadline-perf_counter()-.5)
        if seconds <= 0 or stop.is_set():
            report('stopped', status='INTERRUPTED')
            return
        refined = strong.separate(physical, base, seconds, .01, callback=callback)
        proof_hash = matrix_hash(refined.model)
        report('prepared', separation_status=refined.status, cuts=len(refined.cuts))
        if np.isfinite(refined.lower):
            report('bound', lower=float(refined.lower), origin='continuous_separation')
        seconds = deadline-perf_counter()-.5
        if seconds <= 0 or stop.is_set():
            report('stopped', status='INTERRUPTED')
            return
        result = strong.certify(refined.model, seconds, callback=callback)
        report('finished', status=result.status, lower=result.lower if np.isfinite(result.lower) else None,
               solver_wall_seconds=result.elapsed_seconds, optimizer_calls=1)
    except Exception as error:
        try:
            sender.send(dict(event='failed', message=type(error).__name__,
                             elapsed_seconds=perf_counter()-started))
        except (BrokenPipeError, OSError):
            pass
    finally:
        sender.close()


class StrengthenedCertification:
    """Certify the fixed cold model; only numerical bounds cross back."""
    def __init__(self, whole):
        self.whole, self.o, self.result = whole, whole.o, whole.result
        self.process = self.connection = self.stop = None
        self.finished = False
        whole.diagnostics['strengthened_proof'] = dict(status='NOT_STARTED', bounds=[], upper_feedback=False)

    def update_bound(self, lower):
        if np.isfinite(lower):
            candidate = max(self.result.lower_bound, float(lower))
            if candidate > self.result.objective + 1e-5:
                raise RuntimeError("Certified lower bound exceeds validated upper bound")
            self.result.lower_bound = candidate

    def converged(self):
        lower, upper = self.result.lower_bound, self.result.objective
        if lower > upper + 1e-5:
            raise RuntimeError("Certified lower bound exceeds validated upper bound")
        return (np.isfinite(lower) and np.isfinite(upper) and self.result.validation is not None
                and self.result.validation["feasible"]
                and upper-lower <= 1e-5 + self.o.tolerance*max(1., abs(upper)))

    def start(self):
        if self.process is not None:
            return
        context = mp.get_context('spawn')
        self.connection, sender = context.Pipe(duplex=False)
        self.stop = context.Event()
        self.process = context.Process(target=_proof, daemon=True, kwargs=dict(
            sender=sender, stop=self.stop, d=self.whole.d, scenarios=self.whole.scenarios,
            deadline=self.whole.deadline, started=self.whole.started))
        try:
            self.process.start()
        finally:
            sender.close()

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
