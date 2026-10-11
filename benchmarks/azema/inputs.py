"""Validate exported author inputs against the untouched NC4 and scenario table."""
import json
import os
from pathlib import Path

import numpy as np

from . import COMMIT, JULIA_VERSION, ROOT, RUNTIME
from .artifacts import save, sha256


def julia_command(job):
    runtime = json.loads((RUNTIME/'runtime.json').read_text(encoding='utf-8'))
    return [runtime['julia'], f'--project={RUNTIME / "julia-env"}', '--startup-file=no',
            str(ROOT/'benchmarks/azema/author.jl'), str(Path(job).resolve())]


def julia_environment():
    return dict(os.environ, JULIA_DEPOT_PATH=str(RUNTIME/'julia-depot'), JULIA_NUM_THREADS='1',
                JULIA_NUM_PRECOMPILE_TASKS='1', JULIA_PKG_PRECOMPILE_AUTO='0', GKSwstype='100')


def verify_sources():
    manifest = json.loads((RUNTIME/'source_manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT:
        raise ValueError('Wrong author revision')
    for name, digest in manifest['files'].items():
        if sha256(RUNTIME/'author'/name) != digest:
            raise ValueError(f'Author source changed: {name}')
        if name.startswith('src/') and sha256(RUNTIME/'julia-env'/name) != digest:
            raise ValueError(f'Executable author source changed: {name}')
    for row in manifest['inputs']:
        if sha256(RUNTIME/'author'/row['path']) != row['sha256']:
            raise ValueError(f'Raw input changed: {row["path"]}')
    return manifest


def export_inputs(directory, base=1, batch=1, scenarios=100):
    if base not in range(1, 6) or batch not in (1, 2, 3) or not 1 <= scenarios <= 100:
        raise ValueError('Unknown SMS base, batch or scenario count')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    verify_sources()
    job = directory/'export_job.json'
    save(job, dict(action='export', author_root=str(RUNTIME/'author'), base=base, batch=batch,
                   scenarios=scenarios, output=str(directory/'author_export.json')))
    from .process import run_process
    state = run_process(julia_command(job), directory/'export', 1800, env=julia_environment())
    if state['status'] != 'FINISHED':
        from .controller import Stopped
        raise Stopped(state, 'input_alignment')
    exported = json.loads((directory/'author_export.json').read_text(encoding='utf-8'))
    verify_export(exported, base, batch, scenarios)
    save(directory/'units.json', exported['units'])
    save(directory/'scenarios.json', exported['scenarios'])
    save(directory/'input_audit.json', dict(passed=True, commit=COMMIT, base=base, batch=batch,
        scenarios=scenarios, source_files=13, fields_compared=True,
        author_export_sha256=sha256(directory/'author_export.json'),
        units_sha256=sha256(directory/'units.json'), scenarios_sha256=sha256(directory/'scenarios.json')))
    return directory/'units.json', directory/'scenarios.json'


def verify_export(exported, base, batch, scenarios):
    from netCDF4 import Dataset
    fields = dict(pmin='MinPower', pmax='MaxPower', ramp_up='DeltaRampUp', ramp_down='DeltaRampDown',
                  startup_cost='StartUpCost', cost_b='LinearTerm', cost_c='ConstTerm',
                  initial_output='InitialPower', initial_duration='InitUpDownTime', min_up='MinUpTime',
                  min_down='MinDownTime', source_cost_a='QuadTerm')
    with Dataset(RUNTIME/'author/Data/T-Ramp'/f'100_0_{base}_w.nc4') as nc:
        block = next(iter(nc.groups.values()))
        thermal = sorted(((int(name.split('_')[-1]), group) for name, group in block.groups.items()
                          if getattr(group, 'type', '') == 'ThermalUnitBlock'))
        if len(thermal) != 100 or [i for i, _ in thermal] != list(range(100)):
            raise ValueError('Expected 100 thermal units in original order')
        for target, source in fields.items():
            expected = [float(group[source][:].ravel()[0]) for _, group in thermal]
            np.testing.assert_array_equal(exported['units'][target], expected)
        demand = np.asarray(block['ActivePowerDemand'][:], float)
    units = exported['units']
    assert units['model_profile'] == 'azema_sms' and units['n_units'] == 100 and units['horizon'] == 24
    assert units['shedding_cost'] == units['curtailment_cost'] == 700
    assert not any(units['cost_a']) and not any(units['shutdown_cost'])
    np.testing.assert_array_equal(units['pstart'], units['pmin'])
    np.testing.assert_array_equal(units['pshut'], units['pmin'])
    np.testing.assert_array_equal(units['base_load'], demand)
    errors = np.loadtxt(RUNTIME/'author/Data/Uncertainty/training_set.csv', delimiter=',')
    assert exported['scenarios']['reserve_fraction'] == 0
    assert len(exported['scenarios']['scenarios']) == scenarios
    for k, s in enumerate(exported['scenarios']['scenarios']):
        day = (batch-1)*100+k
        expected = demand*(1+errors[day*24:(day+1)*24, 0])
        np.testing.assert_allclose(s['load'], expected, rtol=1e-14, atol=1e-10)
        assert s['name'] == f'day-{day+1}' and s['probability'] == 1/scenarios
