"""One 1800-second cold SA+LLM run with the revised prompt and async scheduler."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
from suc_benders.benders import dispatch
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.validation import pack, transition_violations, unpack, validate


def save(path, value):
    path.write_text(json.dumps(_serializable(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def hashes(paths):
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Evidence:
    def __init__(self, folder):
        self.proof_directory = folder/'proof'
        self.trace = (folder/'curve.jsonl').open('w', encoding='utf-8', buffering=1)
        self.last = None

    def __call__(self, whole, event):
        value = (whole.result.objective, whole.result.lower_bound)
        llm_event = event in ('llm_request', 'llm_call_status', 'llm_response')
        if value != self.last or llm_event or event in ('initialization', 'round', 'final'):
            row = dict(event=event, seconds=perf_counter()-whole.started,
                upper=value[0], lower=value[1], sa_calls=whole.diagnostics['sa_calls'],
                llm_calls=whole.result.llm_calls)
            if llm_event and whole.result.llm_history:
                call = whole.result.llm_history[-1]
                row['call'] = {k: call.get(k) for k in ('call_id', 'status', 'request_upper',
                    'upper_before_probes', 'upper_after_probes', 'direct_incumbent_decrease', 'request_directory')}
                save(self.proof_directory.parent/'llm-progress.json', whole.result.llm_history)
            self.trace.write(json.dumps(_serializable(row), allow_nan=False)+'\n')
            self.last = value


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    output = ROOT/'results'/f'rcuc20-llm-rerun-{stamp}-{uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT/'data/units.json', ROOT/'data/stochastic.json']
    sources = [*sorted((ROOT/'suc_benders').glob('*.py')), ROOT/'run_suc.py', Path(__file__).resolve()]
    input_hash, source_hash = hashes(inputs), hashes(sources)
    options = dict(data_file=str(inputs[0]), scenario_file=str(inputs[1]), seed=17,
        time_limit=1800, tolerance=.001, block_hours=3, max_outer=1000000, max_inner=20,
        enhanced_search=True, llm_enabled=True, llm_model='gpt-6-astra', llm_codex_command='codex',
        llm_agent_directory=str(output/'llm_requests'), llm_max_calls=3, llm_patience=3,
        llm_timeout=1200., llm_reserve_seconds=60., sa_reads=16, sa_sweeps=5,
        sa_temperature=100., sa_cooling=.98, sa_cutoff=.01)
    manifest = dict(scope='ONE_COLD_RUN_WITH_HISTORICAL_COMPARISONS', attempts=1,
        cold_start=True, forbidden_imports=['historical_commitments', 'pilot_candidates', 'cuts', 'bounds', 'reference_solution'],
        status='RUNNING', output=str(output), input_sha256=input_hash, source_sha256=source_hash,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(output/'options.json', options); save(output/'protocol.json', manifest)
    with zipfile.ZipFile(output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources+inputs:
            archive.write(path, path.relative_to(ROOT))
    print(f'OUTPUT={output}', flush=True)
    d, scenarios = load_data(*inputs)
    assert (d.n_units, d.horizon, len(scenarios)) == (20, 24, 1)
    evidence = Evidence(output)
    try:
        result = run_suc(options, progress=evidence)
    finally:
        evidence.trace.close()
    diagnostic = result.diagnostics
    report = validate(d, scenarios, result) if result.validation is not None else None
    source_gains = {}
    for row in diagnostic['evaluations']:
        if row['improved'] and np.isfinite(row['incumbent_before']):
            source_gains[row['source']] = source_gains.get(row['source'], 0.)+row['incumbent_before']-row['incumbent']
    assert abs(sum(source_gains.values())-(diagnostic['initial_upper']-result.objective)) < .01
    summary = dict(manifest)
    summary.update(status=result.status, message=result.message, elapsed_seconds=result.elapsed_seconds,
        upper=result.objective, lower=result.lower_bound, gap_percent=100*result.relative_gap,
        target_met=bool(report and result.status == 'CONVERGED' and result.relative_gap <= .001 and result.elapsed_seconds <= 1800),
        independent_validation=report, initialization_mip_used='initialization_repair' in diagnostic,
        initial_upper=diagnostic['initial_upper'], improvement_by_source=source_gains,
        llm_calls=result.llm_calls, sa_calls=diagnostic['sa_calls'], block_sa_calls=diagnostic['block_sa_calls'],
        joint_sa_calls=diagnostic['joint_sa_calls'], search_statistics=diagnostic['search_statistics'],
        timings=diagnostic['timings'], llm_background_seconds=diagnostic['llm_background_seconds'],
        model_fingerprint=diagnostic['model_fingerprint'], strengthened_proof=diagnostic['strengthened_proof'])
    save(output/'summary.json', summary)
    save(output/'llm-history.json', result.llm_history)
    print(json.dumps(dict(status=result.status, upper=result.objective, lower=result.lower_bound,
        gap_percent=100*result.relative_gap, llm_calls=result.llm_calls, improvement_by_source=source_gains)), flush=True)
    # Post-run raw-candidate audit never feeds a candidate or bound back into the run.
    physical = model(d, scenarios); full = partition(physical, d.horizon)[0][0]
    fixed = np.array(diagnostic['reachability']['fixed_u'])
    known = fixed != -1
    audit = []
    for call in result.llm_history:
        for proposal in call['proposals']:
            if 'raw_u' not in proposal:
                continue
            u = np.asarray(proposal['raw_u'], float)
            row = dict(call_id=call['call_id'], proposal_id=proposal['proposal_id'],
                fixed_state_violations=int(np.count_nonzero(u[known] != fixed[known])),
                violations=transition_violations(d, u), repair_bits=proposal.get('hamming_u'),
                online_outcome=proposal.get('outcome'), direct_decrease=proposal.get('incumbent_decrease', 0.),
                objective=None, validation=None)
            raw = dispatch(full, pack(d, u), 30.)
            row['raw_status'] = raw.status
            if raw.feasible:
                plan = unpack(physical, raw.point)
                row.update(objective=plan.objective, validation=validate(d, scenarios, plan),
                    decrease_vs_request=call['request_upper']-plan.objective,
                    decrease_vs_receipt=call.get('upper_before_probes', call['request_upper'])-plan.objective)
            audit.append(row)
    save(output/'llm-raw-audit.json', audit)
    assert hashes(inputs) == input_hash and hashes(sources) == source_hash
    save(output/'result.json', asdict(result))
    save(Path(__file__).with_name('summary.json'), summary)
    print('SAVED_ALL_ARTIFACTS', flush=True)
    return int(result.status.endswith('FAILED') or result.status == 'PAUSED_LLM_FAILURE' or report is None)


if __name__ == '__main__':
    raise SystemExit(main())
