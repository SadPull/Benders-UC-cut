"""One revised-prompt call on historical call 2's frozen request; no SA or MIP."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter, sleep
from types import SimpleNamespace as NS
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_suc import _serializable
from suc_benders import llm
from suc_benders.advisor_request import AdvisorRequest
from suc_benders.benders import dispatch
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.solver import WholeDay, Result
from suc_benders.validation import pack, commitment, commitment_hash, repair, unpack, validate, transition_violations

SOURCE = ROOT/'results/manual-seed17-20261017-131535-dbef3176/llm_requests/02-c528f815bccb401e8ced8021c411f152'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(_serializable(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(candidates, d, scenarios, physical, full, baseline, fixed, known):
    cache, rows = {}, []
    deadline = perf_counter()+120.
    def evaluate(x):
        identity = commitment_hash(commitment(d, x))
        if identity in cache:
            return deepcopy(cache[identity])
        remaining = deadline-perf_counter()
        if remaining <= 0:
            return dict(status='UNVERIFIED_TIME_LIMIT', feasible=False)
        r = dispatch(full, x, min(30., remaining))
        result = dict(identity=identity, status=r.status, feasible=r.feasible,
                      duplicate_of_input=identity in known, objective=None, decrease=None)
        if r.feasible:
            plan = unpack(physical, r.point)
            result.update(objective=plan.objective, decrease=baseline-plan.objective,
                          validation=validate(d, scenarios, plan), u=plan.u.tolist())
        cache[identity] = result
        return deepcopy(result)
    for k, candidate in enumerate(candidates):
        u = np.asarray(candidate['u'], float)
        assert u.shape == (d.n_units, d.horizon) and np.isin(u, [0, 1]).all()
        row = dict(candidate=k+1, name=candidate.get('name'), reason=candidate.get('reason'),
                   raw_u=u.tolist(), violations=transition_violations(d, u),
                   fixed_state_violations=int(np.count_nonzero(u[np.isfinite(fixed)] != fixed[np.isfinite(fixed)])))
        row['raw'] = evaluate(pack(d, u))
        x = repair(d, scenarios, u)
        row['repaired'] = (dict(status='REPAIR_FAILED', feasible=False) if x is None else evaluate(x))
        row['repair_bits'] = None if x is None else int(np.count_nonzero(commitment(d, x).u != u))
        rows.append(row)
    summary = dict(proposals=len(rows), fixed_state_violations=sum(r['fixed_state_violations'] for r in rows))
    for stage in ('raw', 'repaired'):
        valid = [r[stage] for r in rows if r[stage]['feasible']]
        new = {r['identity']:r for r in valid if not r['duplicate_of_input']}
        summary[stage] = dict(feasible=len(valid), new_unique_feasible=len(new),
            improving=sum(r['objective'] < baseline-1e-4 for r in new.values()),
            best_objective=min((r['objective'] for r in valid), default=None),
            best_decrease=max([0., *(r['decrease'] for r in new.values())]))
    return dict(summary=summary, candidates=rows)


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    output = ROOT/'results'/f'rcuc20-prompt-pilot-{stamp}-{uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    frozen = read(SOURCE/'request.json')
    old = json.loads(frozen['request']['input'])
    evidence = read(SOURCE.parent.parent/'run-evidence.json')
    for name in ('data/units.json', 'data/stochastic.json'):
        assert digest(ROOT/name) == evidence['input_sha256'][name], 'Frozen input mismatch'
    d, scenarios = load_data(ROOT/'data/units.json', ROOT/'data/stochastic.json')
    for key in ('pmin', 'pmax', 'pstart', 'pshut', 'ramp_up', 'ramp_down', 'initial_on', 'initial_output',
                'min_up', 'min_down', 'initial_duration', 'cost_a', 'cost_b', 'cost_c', 'startup_cost', 'shutdown_cost'):
        np.testing.assert_array_equal(getattr(d, key), old[key])
    for s, saved in zip(scenarios, old['scenarios']):
        np.testing.assert_array_equal(s.load, saved['load'])
        assert s.probability == saved['probability']
    assert len(scenarios) == len(old['scenarios']) and d.reserve_fraction == old['reserve_fraction']
    options = load_options(dict(llm_enabled=True, llm_model=frozen['requested_model'],
        llm_codex_command='codex', llm_agent_directory=str(output/'request'), verbose=False))
    physical = model(d, scenarios); full = partition(physical, d.horizon)[0][0]
    start = perf_counter()
    whole = WholeDay(physical, full, options, start+60., Result(), start)
    incumbent = np.asarray(old['incumbent_u'], float)
    assert whole.accept(pack(d, incumbent), 'frozen_initial')
    baseline = whole.result.objective
    assert abs(baseline-old['context']['incumbent_cost']['total']) < .001
    context = deepcopy(old['context'])
    context['reachability'] = whole.diagnostics['reachability']
    bases = [NS(u=np.asarray(u)) for u in old['base_u']]
    body = llm.request_body(d, scenarios, whole.result, options, context, bases, old['iteration'])
    prompt = json.loads(body['input'])
    for key in ('incumbent_u', 'base_u', 'scenarios', 'pmin', 'pmax', 'ramp_up', 'ramp_down'):
        assert prompt[key] == old[key]
    assert prompt['context']['previous_call_feedback'] == old['context']['previous_call_feedback']
    assert body['model'] == frozen['requested_model'] and body['reasoning']['effort'] == frozen['reasoning_effort']
    known = {commitment_hash(commitment(d, pack(d, u))) for u in [incumbent, *[b.u for b in bases]]}
    fixed = whole.fixed[full.u]
    old_candidates = read(SOURCE/'reply.json')['candidates']
    old_evaluation = assess(old_candidates, d, scenarios, physical, full, baseline, fixed, known)
    save(output/'historical-evaluation.json', old_evaluation)
    paths = [ROOT/'data/units.json', ROOT/'data/stochastic.json', SOURCE/'request.json', SOURCE/'reply.json']
    sources = [*sorted((ROOT/'suc_benders').glob('*.py')), Path(__file__).resolve()]
    manifest = dict(scope='ONE_FROZEN_REQUEST_NOT_COLD_START_OR_GENERAL_LLM_PROOF',
        status='RUNNING', calls=1, retry=False, source=str(SOURCE), output=str(output), baseline=baseline,
        model=body['model'], reasoning=body['reasoning'], timeout_seconds=1200,
        hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths+sources},
        reference_solution_provided=False, future_replies_provided=False,
        historical_old_prompt_rerun=False, sa_calls=0, integer_search_calls=0,
        decommit_filter=prompt['context']['decommit_filter'], reachability=prompt['context']['reachability'])
    save(output/'protocol.json', manifest)
    print(f'OUTPUT={output}', flush=True)
    print(json.dumps(dict(baseline=baseline, historical=old_evaluation['summary'],
                         decommit_filter=manifest['decommit_filter'])), flush=True)
    task = AdvisorRequest(d, options, body, 2, 1200.)
    try:
        reply = None
        while reply is None:
            reply = task.poll()
            if reply is None:
                sleep(.25)
    finally:
        task.close()
    proposals, receipt = reply
    save(output/'receipt.json', receipt)
    forbidden = []
    for line in (task.codex_directory/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        item = event.get('item', {})
        if item and item.get('type') not in ('reasoning', 'agent_message', 'error'):
            forbidden.append(item.get('type'))
    manifest.update(status=receipt['status'], forbidden_items=forbidden,
                    seconds=receipt.get('background_seconds'), thread_id=receipt.get('thread_id'))
    if receipt['status'] == 'OK' and not forbidden:
        candidates = read(task.codex_directory/'reply.json')['candidates']
        assert len(candidates) == len(proposals) <= 4
        evaluated = assess(candidates, d, scenarios, physical, full, baseline, fixed, known)
        save(output/'new-evaluation.json', evaluated)
        manifest.update(evaluation=evaluated['summary'], historical=old_evaluation['summary'],
            raw_improvement_vs_historical=evaluated['summary']['raw']['best_decrease']-old_evaluation['summary']['raw']['best_decrease'],
            repaired_improvement_vs_historical=evaluated['summary']['repaired']['best_decrease']-old_evaluation['summary']['repaired']['best_decrease'])
    elif forbidden:
        manifest['status'] = 'INVALID_TOOL_USE'
    assert all(digest(ROOT/name) == value for name,value in manifest['hashes'].items())
    save(output/'summary.json', manifest)
    save(Path(__file__).with_name('summary.json'), manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('hashes', 'reachability')}, ensure_ascii=True), flush=True)
    return 0 if manifest['status'] == 'OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
