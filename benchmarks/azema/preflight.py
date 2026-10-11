"""Numerical gates. Called only by the memory-gated isolated worker."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from suc_benders.anneal import initialize, anneal
from suc_benders.benders import coordinate, dispatch, master
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.qubo import qubo
from suc_benders.solver import BlockSearch, Result, WholeDay
from suc_benders.validation import pack, unpack, validate
from . import RUNTIME
from .artifacts import Recorder, save
from .inputs import julia_command, julia_environment


def tiny_cases():
    # Independent fixed expectations include author-specific t=0 and t=T semantics.
    d = dict(model_profile='azema_sms', n_units=1, horizon=3, pmin=[10.], pmax=[55.],
        pstart=[10.], pshut=[10.], ramp_up=[15.], ramp_down=[10.], min_up=[1], min_down=[1],
        initial_duration=[-3], initial_output=[0.], cost_a=[0.], source_cost_a=[.02],
        cost_b=[2.], cost_c=[5.], startup_cost=[3.], shutdown_cost=[0.],
        base_load=[30.]*3, shedding_cost=700., curtailment_cost=700.)
    definitions = [
        ('startup_and_terminal_shutdown', {}, [[1, 1, 0]], [30., 30., 0.]),
        ('initial_ramp', dict(initial_duration=[5], initial_output=[25.]), [[1, 1, 0]], [30., 30., 0.]),
        ('minimum_downtime', dict(min_down=[3], initial_duration=[-1]), [[0, 0, 1]], [30.]*3),
        ('minimum_uptime', dict(min_up=[3], initial_duration=[1], initial_output=[10.]), [[1, 1, 0]], [0.]*3),
        ('zero_initial_duration', dict(min_up=[3], initial_duration=[0], initial_output=[10.]), [[0, 0, 0]], [0.]*3),
        ('shedding', {}, [[0, 0, 0]], [120.]*3),
        ('curtailment', dict(min_up=[3], initial_duration=[1], initial_output=[10.]), [[1, 1, 1]], [0.]*3)]
    for name, changes, fixed, loads in definitions:
        units = dict(deepcopy(d), **changes)
        scenarios = dict(reserve_fraction=0, scenarios=[
            dict(name='low', probability=.5, load=loads),
            dict(name='high', probability=.5, load=[x+5 for x in loads])])
        yield name, units, scenarios, fixed


def author_check(directory, units, scenarios, mode, u=None, seconds=120, all_scenarios=False):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    task = dict(action='audit' if all_scenarios else 'check', author_root=str(RUNTIME/'author'), units=units,
        scenarios=scenarios, mode=mode, time_limit=seconds, output=str(directory/'author.json'))
    if u is not None:
        task['u'] = np.asarray(u).tolist()
    save(directory/'job.json', task)
    with (directory/'author.log').open('w', encoding='utf-8') as log:
        subprocess.run(julia_command(directory/'job.json'), env=julia_environment(),
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=seconds+120,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return json.loads((directory/'author.json').read_text(encoding='utf-8'))


def cross_language(job):
    rows = []
    for name, units, scenarios, fixed in tiny_cases():
        folder = Path(job['directory'])/name
        save(folder/'units.json', units); save(folder/'scenarios.json', scenarios)
        d, ss = load_data(folder/'units.json', folder/'scenarios.json')
        m = model(d, ss); full = partition(m, d.horizon)[0][0]
        for mode in ('fixed', 'relaxation', 'integer'):
            if mode == 'fixed':
                value = dispatch(full, pack(d, np.asarray(fixed)), 60, certified=True)
                if not value.feasible:
                    raise AssertionError(f'Test schedule unexpectedly infeasible: {name}')
                validate(d, ss, unpack(m, value.point))
                cost = value.cost
            else:
                value = optimize(m, 60, integral=mode == 'integer')
                if value.status != 'OPTIMAL':
                    raise AssertionError(f'Unsolved Python comparison: {name}/{mode}')
                cost = value.objval
            reference = author_check(folder/mode, units, scenarios, mode, fixed)
            if reference['status'] != 'OPTIMAL' or abs(reference['objective']-cost) > 1e-4:
                raise AssertionError(f'Author/Python mismatch: {name}/{mode}')
            rows.append(dict(case=name, mode=mode, python=cost, author=reference['objective'], passed=True))
            save(Path(job['directory'])/'comparisons.json', rows)
    save(Path(job['directory'])/'check.json', dict(passed=True, comparisons=rows))


def initialized_whole(job, *, certification):
    from .worker import read_common
    started = perf_counter()
    options = load_options(job['options'])
    d, scenarios = load_data(options.data_file, options.scenario_file)
    m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
    full.audit_cuts = True
    # Preflight is independently timed; no preflight cuts or candidates enter a formal run.
    options.time_limit = job['time_limit']
    options.bound_mode = 'integer_benders' if certification else 'relaxation'
    recorder = Recorder(job['directory'], options.time_limit)
    whole = WholeDay(m, full, options, started+options.time_limit, Result(), started, recorder)
    common = read_common(job['common_directory'])
    if common['model_fingerprint'] != whole.diagnostics['model_fingerprint']:
        raise ValueError('Preflight common input mismatch')
    whole.result.lower_bound = common['lower_bound']
    if not whole.accept(pack(d, common['u']), 'initial', {'proposal_id': 'preflight:initial'}):
        raise RuntimeError('Preflight initial reevaluation failed')
    return whole, common['relaxation_prices']


def common_audit(job):
    from .worker import read_common
    from .audit import checked_author_schedule
    directory, inputs = Path(job['directory']), Path(job['inputs'])
    common = read_common(job['common_directory'])
    d, scenarios = load_data(inputs/'units.json', inputs/'scenarios.json')
    m = model(d, scenarios)
    from suc_benders.model import fingerprint
    if fingerprint(m) != common['model_fingerprint']:
        raise ValueError('Cached initialization does not match the current complete model')
    prices = common['relaxation_prices']
    if prices.shape != m.rhs.shape or (prices[~m.eq] < 0).any() or not np.isfinite(prices).all():
        raise ValueError('Invalid common relaxation dual certificate')
    c = m.obj+np.asarray(m.A.T @ prices)
    lower = float(np.minimum(c*m.lb, c*m.ub).sum()-m.rhs @ prices)
    if common['lower_bound'] > lower+1e-5:
        raise ValueError('Common lower bound exceeds independently recomputed dual value')
    reference = author_check(directory/'author',
        json.loads((inputs/'units.json').read_text(encoding='utf-8')),
        json.loads((inputs/'scenarios.json').read_text(encoding='utf-8')),
        'fixed', common['u'], seconds=120, all_scenarios=True)
    validation = checked_author_schedule(d, scenarios, reference, common['u'], common['objective'])
    save(directory/'check.json', dict(passed=True, validation=validation,
        independent_lower_bound=lower, scenarios=len(scenarios)))


def real_sa(job):
    whole, prices = initialized_whole(job, certification=False)
    initialize(whole.o.kaiwu_user_id, whole.o.kaiwu_sdk_code)
    blocks, cross = partition(whole.m, 3, prices)
    searches = [BlockSearch(b, whole.o, 1, j+1, whole) for j, b in enumerate(blocks)]
    if len(searches) != 8:
        raise ValueError('Real SA preflight requires all eight original-size blocks')
    for search in searches:
        b = search.b
        # Same seeds, master, QUBO, production sampler, repair and global evaluation as BlockSearch.step.
        search.evaluate(whole.best.point[b.cols[b.binary]], 'initial')
        mp, bound = master(b, search.cuts, whole.remaining())
        mp.lb[-1] = max(mp.lb[-1], bound-b.constant)
        encoded = qubo(mp, whole.o, search.upper-b.constant)
        whole.diagnostics['pending_qubo'] = dict(block=search.index, bits=encoded.Q.shape[0],
            nonzeros=encoded.Q.nnz, max_coefficient=float(abs(encoded.Q).max()),
            dense_working_bytes=32*(encoded.Q.shape[0]+1)**2)
        whole.emit('before_sampling')
        whole.diagnostics.setdefault('qubo_checks', []).append(whole.diagnostics['pending_qubo'].copy())
        whole.diagnostics['sa_calls'] += 1
        proposals, sampling = anneal(encoded, whole.o, whole.remaining(), whole.o.seed)
        sampling.update(block=search.index, seed=whole.o.seed)
        whole.diagnostics['sampling'].append(sampling)
        if not proposals or whole.remaining() <= 0:
            raise RuntimeError('Real SA preflight returned no timely candidates')
        baseline = whole.result.u.copy()
        for index, candidate in enumerate(proposals):
            local = np.zeros(len(b.obj)); local[b.binary] = candidate
            raw = baseline.copy(); raw[:, b.times] = local[b.u]
            record = dict(proposal_id=f'preflight-sa:{search.index}:{index}')
            repaired = whole.prepare(raw, 'sa', record)
            if repaired is not None:
                search.evaluate(repaired[b.cols[b.binary]], 'sa')
                whole.accept(repaired, 'sa', record)
        whole.emit('sampling')
    values, upper = coordinate([s.planes for s in searches], whole.m.eq[cross],
                               whole.result.objective, whole.remaining())
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite coordination multipliers')
    # Re-encode every block at the coordinated prices to catch the historical large-price failure.
    coordinated = np.zeros(len(whole.m.rhs)); coordinated[cross] = values
    for block in partition(whole.m, 3, coordinated)[0]:
        evaluation = dispatch(block, whole.best.point[block.cols[block.binary]], whole.remaining())
        mp, bound = master(block, [evaluation.cut], whole.remaining())
        mp.lb[-1] = max(mp.lb[-1], bound-block.constant)
        encoded = qubo(mp, whole.o, evaluation.cost-block.constant)
        if not np.isfinite(encoded.Q.data).all():
            raise RuntimeError('Nonfinite QUBO after coordination')
    whole.diagnostics['coordination_count'] = 1
    whole.emit('coordination')
    whole.emit('final')
    save(Path(job['directory'])/'check.json', dict(passed=True, blocks=8, real_sa_calls=8,
        coordination_count=1, max_multiplier=float(abs(values).max(initial=0)),
        qubo=whole.diagnostics['qubo_checks'], llm_calls=whole.result.llm_calls))


def real_llm(job):
    whole, _ = initialized_whole(job, certification=True)
    try:
        whole.certifier.step()
        if whole.certifier.last_master is None:
            raise RuntimeError('No early integer-master probe for lower-channel preflight')
        for channel in ('upper', 'lower'):
            # Require fresh numerical evaluation even if the service repeats the initial schedule.
            whole.cache.clear()
            whole.advice.begin([channel], validation=True)
            whole.advice.run_batch([])
        receipts = whole.result.llm_history
        for channel in ('upper', 'lower'):
            if not any(r['channel'] == channel and r['status'] == 'OK'
                       and r.get('numerical_evaluation_completed') for r in receipts):
                raise RuntimeError(f'No real, numerically evaluated {channel} reply')
        whole.emit('final')
        save(Path(job['directory'])/'check.json', dict(passed=True, channels=['upper', 'lower'],
            llm_calls=whole.result.llm_calls, model=whole.o.llm_model, reasoning='max',
            formal_calls_counted=0))
    finally:
        whole.advice.close()
        whole.certifier.close()
