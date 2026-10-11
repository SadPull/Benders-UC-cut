"""RTS-24 inputs, continuous dispatch checks and the independent reference."""
import json
from pathlib import Path

import numpy as np

from suc_benders.data import load_data
from suc_benders.model import fingerprint, model
from suc_benders.optimization import optimize
from suc_benders.validation import pack, unpack, validate
from . import ROOT
from .artifacts import save, sha256

RTS24 = ROOT/'data/rts24'
INPUTS = ('units.json', 'stochastic.json', 'provenance.json')


def quadratic_dual(m, prices):
    prices = np.asarray(prices)
    if prices.shape != m.rhs.shape or not np.isfinite(prices).all() or (prices[~m.eq] < 0).any():
        raise ValueError('Invalid initial dual certificate')
    c = m.obj+np.asarray(m.A.T @ prices)
    z = np.where(c < 0, m.ub, m.lb).copy()
    curved = m.q > 0
    z[curved] = np.clip(-c[curved]/(2*m.q[curved]), m.lb[curved], m.ub[curved])
    return float(np.sum(c*z+m.q*z*z)-m.rhs @ prices)


def case(inputs):
    d, scenarios = load_data(inputs/'units.json', inputs/'stochastic.json')
    if (d.n_units, d.horizon, len(scenarios), d.reserve_fraction) != (32, 24, 3, .05):
        raise ValueError('RTS-24 input shape or reserve changed')
    return model(d, scenarios)


def fixed_check(m, u, expected, seconds):
    x = pack(m.d, np.asarray(u, float))
    m.lb[m.binary] = x; m.ub[m.binary] = x
    result = optimize(m, seconds)
    if result.status != 'OPTIMAL':
        raise ValueError('Fresh fixed-commitment QP did not solve optimally')
    solved = unpack(m, result.x)
    validation = validate(m.d, m.scenarios, solved)
    if abs(result.objval-expected) > 1e-4:
        raise ValueError('Fresh QP cost differs from recorded cost')
    return dict(objective=result.objval, validation=validation, scenarios=solved.scenarios,
        fixed_cost=result.objval-sum(s.probability*z.objective for s, z in zip(m.scenarios, solved.scenarios)))


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def reference_check(job):
    m = case(Path(job['inputs']))
    manifest = read(RTS24/'reference/manifest.json'); reference = read(RTS24/'reference/result.json')
    if manifest['model_fingerprint'] != fingerprint(m) or manifest['input_sha256'] != {
        name: sha256(Path(job['inputs'])/name) for name in INPUTS}:
        raise ValueError('Reference inputs changed')
    checked = fixed_check(m, reference['u'], reference['checked']['objective'], job['time_limit'])
    save(Path(job['directory'])/'check.json', dict(passed=True, source=str(RTS24/'reference/result.json'),
        source_sha256=sha256(RTS24/'reference/result.json'), lower_bound=reference['lower_bound'],
        solver_gap=reference['relative_gap'], original_solve_seconds=reference['solve_wall_seconds'], **checked))
