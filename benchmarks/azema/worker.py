"""One isolated common-initialization, preflight or project-arm process."""
from time import perf_counter, time
PROCESS_STARTED = perf_counter()
PROCESS_STARTED_UNIX = time()
import argparse
import json
from pathlib import Path

import gurobipy as gp
import numpy as np

from . import GUROBI_VERSION
from .artifacts import Recorder, save, sha256, memory_gate, source_hashes


def common_initialization(job):
    from suc_benders.data import load_data
    from suc_benders.model import model, partition, fingerprint
    from suc_benders.optimization import optimize
    from suc_benders.benders import dispatch
    from suc_benders.validation import repair, unpack, validate
    from suc_benders.startup import fixed_cost
    started = job.get('started_at', PROCESS_STARTED)
    remaining = lambda: job['time_limit']-(perf_counter()-started)
    directory=Path(job['directory']); directory.mkdir(parents=True,exist_ok=True)
    save(directory/'progress.json',dict(stage='model_construction',elapsed_seconds=perf_counter()-started))
    d, scenarios = load_data(job['data_file'], job['scenario_file'])
    m = model(d, scenarios); b = partition(m, d.horizon)[0][0]
    save(directory/'progress.json',dict(stage='relaxation',method=2,elapsed_seconds=perf_counter()-started))
    relaxation = optimize(m, remaining(), method=2, log_file=directory/'relaxation-gurobi.log')
    if relaxation.status != 'OPTIMAL' or not np.isfinite(relaxation.lower):
        raise RuntimeError('Common relaxation has no optimal certified solution')
    candidates = []
    for raw in (np.floor(relaxation.x[m.ix.u]+.5), np.ones((d.n_units,d.horizon))):
        save(directory/'progress.json',dict(stage='fixed_dispatch',candidate=len(candidates)+1,
            lower_bound=relaxation.lower,elapsed_seconds=perf_counter()-started))
        x = repair(d, scenarios, raw)
        r = dispatch(b, x, remaining(), certified=True)
        if r.feasible:
            z = unpack(m,r.point); validation = validate(d,scenarios,z)
            candidates.append((z.objective,z,validation))
    if not candidates or remaining() <= 0:
        raise RuntimeError('No common initial schedule within initialization budget')
    cost,z,validation = min(candidates,key=lambda row:row[0])
    arrays=directory/'common_arrays.npz'
    np.savez_compressed(arrays,relaxation_prices=relaxation.prices)
    if relaxation.lower > cost+1e-5:
        raise RuntimeError('Common bounds are crossed')
    state=dict(objective=cost, lower_bound=relaxation.lower, u=z.u, validation=validation,
        model_fingerprint=fingerprint(m), fixed_cost=fixed_cost(d,z.u,z.w),
        scenario_recourse=[getattr(s, 'recourse_cost', s.objective) for s in z.scenarios],
        relaxation_method=2,
        arrays_sha256=sha256(arrays), charged_elapsed_seconds=perf_counter()-started)
    save(directory/'common.json',state)
    save(directory/'initialization_source_sha256.json', source_hashes())


def read_common(directory):
    directory=Path(directory)
    state=json.loads((directory/'common.json').read_text(encoding='utf-8'))
    if sha256(directory/'common_arrays.npz') != state['arrays_sha256']:
        raise ValueError('Common initialization array hash mismatch')
    state['relaxation_prices']=np.load(directory/'common_arrays.npz',allow_pickle=False)['relaxation_prices']
    state['u']=np.asarray(state['u'],float)
    state['sha256']=sha256(directory/'common.json')
    return state


def run_project(job):
    from run_suc import run_suc
    started=PROCESS_STARTED-max(0., PROCESS_STARTED_UNIX-job.get('launch_epoch_seconds', PROCESS_STARTED_UNIX))
    initial=read_common(job['common_directory'])
    record=Recorder(job['directory'],job['time_limit'])
    result=run_suc(job['options'],initial_state=initial,progress=record,started_at=started)
    save(Path(job['directory'])/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    job=json.loads(parser.parse_args().job.read_text(encoding='utf-8'))
    directory=Path(job['directory']); directory.mkdir(parents=True,exist_ok=True)
    gate=memory_gate(job.get('estimated_peak_bytes',0))
    save(directory/'resource_gate.json',gate)
    if not gate['passed']:
        save(directory/'state.json',dict(status='BLOCKED_MEMORY',numerical_runs_started=0,llm_calls_started=0))
        return 3
    if '.'.join(map(str,gp.gurobi.version())) != GUROBI_VERSION:
        raise RuntimeError('Use the isolated Gurobi 12.0.2 runtime')
    if job['action']=='common':
        common_initialization(job)
    elif job['action']=='run':
        run_project(job)
    elif job['action']=='crosscheck':
        from .preflight import cross_language
        cross_language(job)
    elif job['action']=='sa':
        from .preflight import real_sa
        real_sa(job)
    elif job['action']=='llm':
        from .preflight import real_llm
        real_llm(job)
    elif job['action']=='audit':
        from .audit import audit_run
        audit_run(job)
    elif job['action']=='common_audit':
        from .preflight import common_audit
        common_audit(job)
    else:
        raise ValueError('Unknown worker action')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
