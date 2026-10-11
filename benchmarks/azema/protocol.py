"""Frozen first-round protocol; credentials are read locally and never serialized."""
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import sys

import gurobipy as gp
import tomli

from suc_benders.config import load_options
from . import COMMIT, GUROBI_VERSION, JULIA_VERSION, ROOT, RUNTIME
from .artifacts import sha256, source_hashes
from .inputs import verify_sources


@dataclass(frozen=True)
class Protocol:
    base_file: str = '100_0_1_w.nc4'
    batch: int = 1
    seed: int = 7
    time_limit: float = 3600.
    scenarios: int = 100
    force: float = 1.
    tolerance: float = .001
    threads: int = 1

    def validate(self):
        if self != Protocol():
            raise ValueError('The approved first round is frozen: base 1, batch 1, seed 7, 100 scenarios, 3600 s')


def options_for(arm, inputs, protocol):
    if arm not in ('B', 'C', 'D'):
        raise ValueError('Project arm must be B, C or D')
    values = dict(data_file=str(Path(inputs)/'units.json'), scenario_file=str(Path(inputs)/'scenarios.json'),
        bound_mode='integer_benders', bound_slice_seconds=30, block_hours=3, max_outer=20, max_inner=20,
        seed=protocol.seed, time_limit=protocol.time_limit, tolerance=protocol.tolerance,
        sa_enabled=arm != 'B', llm_enabled=arm == 'D', llm_patience=3, llm_max_calls=12,
        llm_timeout=600, verbose=True)
    options = load_options(values)
    if arm == 'D' and options.llm_model != 'gpt-6-astra':
        raise ValueError('Arm D requires the configured gpt-6-astra model; reasoning effort is max')
    # All fields, including production SA settings, are frozen in the public job file.
    private = {'kaiwu_user_id', 'kaiwu_sdk_code', 'llm_url', 'llm_api_key'}
    return {k: v for k, v in asdict(options).items() if k not in private}


def environment():
    manifest = tomli.loads((RUNTIME/'julia-env/Manifest.toml').read_text(encoding='utf-8'))
    original = tomli.loads((RUNTIME/'author/Manifest.toml').read_text(encoding='utf-8'))
    for name, packages in original['deps'].items():
        if manifest['deps'].get(name) != packages:
            raise ValueError(f'Pinned Julia dependency changed: {name}')
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'gurobipy', 'kaiwu', 'psutil', 'netCDF4', 'pytest')}
    native = '.'.join(map(str, gp.gurobi.version()))
    if native != GUROBI_VERSION or versions['gurobipy'] != GUROBI_VERSION:
        raise ValueError('Run the benchmark in its isolated Gurobi 12.0.2 environment')
    if manifest['julia_version'] != JULIA_VERSION or manifest['deps']['Gurobi'][0]['version'] != '1.7.4':
        raise ValueError('Julia or Gurobi.jl version mismatch')
    return dict(python=sys.version, executable=sys.executable, packages=versions, gurobi_native=native,
        julia=manifest['julia_version'], julia_gurobi=manifest['deps']['Gurobi'][0]['version'],
        jump=manifest['deps']['JuMP'][0]['version'], ncdatasets=manifest['deps']['NCDatasets'][0]['version'],
        files={name: sha256(RUNTIME/name) for name in ('runtime.json', 'source_manifest.json',
                'julia-env/Project.toml', 'julia-env/Manifest.toml')})


def binding(protocol, inputs):
    upstream = verify_sources()
    public = options_for('D', inputs, protocol)
    return dict(protocol=asdict(protocol), source=source_hashes(), environment=environment(),
        author_commit=COMMIT, upstream=upstream, options=public,
        inputs={name: sha256(Path(inputs)/name) for name in ('units.json', 'scenarios.json', 'author_export.json')})


def require_preflight(path, current):
    evidence = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'input_alignment', 'environment', 'cross_language', 'core_regression', 'cut_enumeration',
                'common_initialization', 'real_sa_eight_blocks', 'real_dual_llm', 'channels_disabled'}
    if evidence.get('status') != 'PASSED' or evidence.get('binding') != current:
        raise ValueError('Preflight missing, incomplete or stale for these inputs, sources or environment')
    if not required.issubset(evidence.get('passed_checks', [])):
        raise ValueError('Preflight did not pass every required check')
    return evidence
