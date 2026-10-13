from dataclasses import asdict
from itertools import product
import json
from time import perf_counter

import numpy as np
import pytest

from suc_benders import solver
from suc_benders.benders import dispatch
from suc_benders.config import load_options
from suc_benders.lower_bound import cover_cut
from suc_benders.model import fingerprint, model, partition
from suc_benders.optimization import optimize
from suc_benders.subagent_advisor import save
from suc_benders.validation import pack


def prepared(tiny, options):
    d, scenarios = tiny()
    options.bound_mode='integer_benders'; options.rts24_improvements=True
    m=model(d,scenarios); full=partition(m,d.horizon)[0][0]; full.audit_cuts=True
    start=perf_counter()
    whole=solver.WholeDay(m,full,options,start+30,solver.Result(lower_bound=0.),start)
    assert whole.accept(pack(d,np.ones((d.n_units,d.horizon))),'initial')
    evidence=dict(verified=True,model_fingerprint=fingerprint(m),source_sha256='a'*64,
                  lower_bound=200.,cover_cuts=[cover_cut(m,t,[]) for t in range(d.horizon)])
    return whole,evidence


@pytest.mark.parametrize('bad',['fingerprint','provenance','unverified','floor','cut'])
def test_inherited_evidence_rejected(tiny,options,bad):
    whole,evidence=prepared(tiny,options)
    try:
        if bad=='fingerprint': evidence['model_fingerprint']='wrong'
        elif bad=='provenance': evidence['source_sha256']=''
        elif bad=='unverified': evidence['verified']=False
        elif bad=='floor': evidence['lower_bound']=whole.result.objective+1
        else: evidence['cover_cuts'][0]['indices']=[0]
        with pytest.raises(ValueError): whole.certifier.inherit(evidence)
    finally: whole.certifier.close()


def test_zero_sa_quota_still_certifies(tiny,options,monkeypatch):
    d,scenarios=tiny()
    options.bound_mode='integer_benders'; options.rts24_improvements=True
    options.sa_budget_seconds=0.
    monkeypatch.setattr(solver,'anneal',lambda *a:pytest.fail('SA quota was exhausted'))
    result=solver.solve(d,scenarios,options)
    assert result.status=='CONVERGED',result.message
    assert result.diagnostics['sa_budget_exhausted']
    assert result.diagnostics['sa_calls']==0
    assert result.certification_history


def test_sa_scope_caps_dispatch_and_charges_all_work(tiny,options,monkeypatch):
    whole,_=prepared(tiny,options); options.sa_budget_seconds=2.
    original=whole.deadline; clock=[perf_counter()]
    monkeypatch.setattr(solver,'perf_counter',lambda:clock[0])
    try:
        with pytest.raises(solver.SABudgetExhausted):
            with whole.sa_work():
                assert whole.remaining()<=2.
                clock[0]+=2.1
                raise TimeoutError
        assert whole.deadline==original
        assert whole.sa_used()==pytest.approx(2.1)
    finally: whole.certifier.close()


def test_warm_start_preserves_lp_provenance_and_resumes_quota(tiny,options,monkeypatch):
    whole,evidence=prepared(tiny,options)
    try:
        lp=optimize(whole.m,5)
        initial=dict(charged_elapsed_seconds=.01,sha256='b'*64,model_fingerprint=fingerprint(whole.m),
            u=whole.result.u,objective=whole.result.objective,lower_bound=lp.lower,
            relaxation_prices=lp.prices,inherited_evidence=evidence)
        options.sa_budget_seconds=0.
        events=[]
        monkeypatch.setattr(solver,'anneal',lambda *a:pytest.fail('No SA budget'))
        result=solver.solve(whole.d,whole.scenarios,options,initial_state=initial,
                            progress=lambda w,e:events.append(e))
        assert result.status=='CONVERGED',result.message
        assert result.diagnostics['inherited_evidence']['lower_bound']==200.
        assert initial['lower_bound']==lp.lower
        saved=asdict(result)
        saved.update(charged_elapsed_seconds=2.,resume_cursor=dict(outer=1,round=1),resume={})
        saved['diagnostics']['sa_budget_seconds_used']=2.
        resumed=solver.solve(whole.d,whole.scenarios,options,resume_state=saved)
        assert resumed.status=='CONVERGED',resumed.message
        assert resumed.elapsed_seconds>=2.
        assert resumed.diagnostics['sa_budget_seconds_used']>=2.
        assert resumed.llm_calls==result.llm_calls
    finally: whole.certifier.close()


@pytest.mark.parametrize('quota',[-1,float('nan'),True])
def test_invalid_sa_quota(tmp_path,quota):
    with pytest.raises(ValueError,match='SA budget'):
        load_options(dict(bound_mode='integer_benders',sa_budget_seconds=quota),tmp_path/'absent')


def test_request_modification_is_rejected(tmp_path,tiny,options):
    from test_subagent_advisor import task
    from suc_benders.agent_bridge import dispatch as register, complete
    advisor=task(tmp_path,tiny,options); folder=advisor.agent.directory
    register(folder/'request.json','agent','upper')
    save(folder/'reply.json',dict(candidates=[dict(u=[[1,1,1]])]))
    complete(folder/'request.json','agent')
    request=json.loads((folder/'request.json').read_text()); request['request']['input']='changed'
    save(folder/'request.json',request)
    proposals,receipt=advisor.poll()
    assert not proposals and receipt['status']=='FAILED'


def test_inherited_covers_bound_every_tiny_feasible_commitment(tiny, options):
    whole, evidence = prepared(tiny, options)
    try:
        whole.certifier.inherit(evidence)
        whole.certifier.step()
        claimed = whole.result.lower_bound
        assert claimed > evidence['lower_bound']
        assert claimed <= 225.+1e-5
        feasible = 0
        for bits in product([0., 1.], repeat=6):
            u = np.asarray(bits).reshape((2, 3))
            r = dispatch(whole.full, pack(whole.d, u), 2, certified=True)
            if r.feasible:
                feasible += 1
                assert all(u[:, t].sum() >= 1 for t in range(3))
                assert r.cost+1e-5 >= claimed
        assert feasible
    finally:
        whole.certifier.close()
