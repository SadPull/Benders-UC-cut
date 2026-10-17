"""Exactly two serial cold starts: 300 s diagnosis, then 1800 s acceptance."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_suc import run_suc, _serializable
from suc_benders.data import load_data
from suc_benders.validation import validate


def save(path, value):
    path.write_text(json.dumps(_serializable(value), ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n', encoding='utf-8')


def hashes(paths):
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


class Evidence:
    def __init__(self, folder):
        self.proof_directory = folder/'proof'
        self.trace = (folder/'curve.jsonl').open('w', encoding='utf-8', buffering=1)
        self.last = None

    def __call__(self, whole, event):
        value = (whole.result.objective, whole.result.lower_bound)
        if value != self.last or event in ('initialization', 'round', 'final'):
            row = dict(event=event, seconds=perf_counter()-whole.started, upper=value[0], lower=value[1],
                       sa_calls=whole.diagnostics['sa_calls'], llm_calls=whole.result.llm_calls)
            self.trace.write(json.dumps(_serializable(row), allow_nan=False)+'\n')
            self.last = value


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    output = ROOT/'results'/f'rcuc20-upper-search-{stamp}-{uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT/'data/units.json', ROOT/'data/stochastic.json']
    sources = [*sorted((ROOT/'suc_benders').glob('*.py')), ROOT/'run_suc.py',
               ROOT/'run_experiment.py', ROOT/'requirements.txt', Path(__file__).resolve(),
               *sorted((ROOT/'tests').glob('*.py'))]
    input_hash, source_hash = hashes(inputs), hashes(sources)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(seed=17, budgets=[300, 1800], llm_enabled=False, cold_start=True,
        forbidden_imports=['historical_commitments', 'cuts', 'bounds', 'reference_solution'],
        input_sha256=input_hash, source_sha256=source_hash, base_commit=commit,
        output=str(output), target_gap_percent=.1, status='RUNNING', runs=[])
    save(output/'manifest.json', manifest)
    with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources+inputs:
            archive.write(path, path.relative_to(ROOT))
    (output/'working.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    print(f'OUTPUT={output}', flush=True)
    d, scenarios = load_data(*inputs)
    assert (d.n_units, d.horizon, len(scenarios)) == (20, 24, 1)
    for seconds in manifest['budgets']:
        assert hashes(inputs) == input_hash and hashes(sources) == source_hash, 'Inputs or code changed between runs'
        folder = output/str(seconds)
        folder.mkdir()
        options = dict(data_file=str(inputs[0]), scenario_file=str(inputs[1]), seed=17,
            time_limit=seconds, tolerance=.001, block_hours=3, max_outer=1000000, max_inner=20,
            enhanced_search=True, llm_enabled=False, sa_reads=16, sa_sweeps=5,
            sa_temperature=100., sa_cooling=.98, sa_cutoff=.01)
        save(folder/'options.json', options)
        evidence = Evidence(folder)
        try:
            result = run_suc(options, progress=evidence)
        finally:
            evidence.trace.close()
        save(folder/'result.json', asdict(result))
        report = validate(d, scenarios, result) if result.validation is not None else None
        diagnostic = result.diagnostics
        summary = dict(status=result.status, message=result.message, seconds=seconds,
            elapsed_seconds=result.elapsed_seconds, upper=result.objective, lower=result.lower_bound,
            gap_percent=100*result.relative_gap, independent_validation=report,
            initialization_mip_used='initialization_repair' in diagnostic,
            llm_calls=result.llm_calls, sa_calls=diagnostic['sa_calls'],
            block_sa_calls=diagnostic['block_sa_calls'], joint_sa_calls=diagnostic['joint_sa_calls'],
            reachability=diagnostic['reachability'], search_statistics=diagnostic['search_statistics'],
            timings=diagnostic['timings'], joint_cuts_added=diagnostic['joint_cuts_added'],
            joint_cuts_reused=diagnostic['joint_cuts_reused'],
            model_fingerprint=diagnostic['model_fingerprint'],
            strengthened_proof=diagnostic['strengthened_proof'],
            input_sha256=input_hash, source_sha256=source_hash,
            accepted=bool(result.status == 'CONVERGED' and report and result.relative_gap <= .001
                          and result.elapsed_seconds <= seconds and 'initialization_repair' not in diagnostic))
        stats = diagnostic['search_statistics']
        summary['correctness_pass'] = bool(report and result.llm_calls == 0
            and not summary['initialization_mip_used']
            and result.status in ('CONVERGED', 'TIME_LIMIT')
            and result.lower_bound <= result.objective+1e-5
            and diagnostic['reachability']['movable_units'] == [1, 4, 7, 9, 14, 17, 18, 19, 20]
            and all(stats[source]['raw_fixed_bit_violations'] == 0 for source in ('sa', 'joint_sa')))
        save(folder/'summary.json', summary)
        manifest['runs'].append(summary)
        save(output/'manifest.json', manifest)
        print(json.dumps(dict(seconds=seconds, status=result.status, upper=result.objective,
            lower=result.lower_bound, gap_percent=100*result.relative_gap,
            correctness_pass=summary['correctness_pass'], accepted=summary['accepted'])), flush=True)
        if not summary['correctness_pass']:
            manifest['status'] = 'CORRECTNESS_FAILED'
            save(output/'manifest.json', manifest)
            return 2
    manifest['status'] = 'ACCEPTED' if manifest['runs'][-1]['accepted'] else 'TARGET_NOT_MET'
    save(output/'manifest.json', manifest)
    save(Path(__file__).with_name('summary.json'), manifest)
    return 0 if manifest['status'] == 'ACCEPTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
