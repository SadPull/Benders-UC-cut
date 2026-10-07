function tests = test_stagnation_online_report
tests=functiontests(localfunctions);
end

function setup(t)
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
fixture=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
t.TestData.parent=fixture.Folder;
t.TestData.directory=fullfile(fixture.Folder,'phase2'); mkdir(t.TestData.directory);
end

function testMissingResultsRemainPending(t)
s=suc.stagnation_online_report(t.TestData.directory);
verifyEqual(t,s.status,'PENDING'); verifyEqual(t,numel(s.runs),6);
verifyTrue(t,all(strcmp({s.runs.attribution_status},'UNKNOWN')));
end

function testWithinBlockCostAttributionAndLedgerWindow(t)
h=[hist(1,1,1,-100),hist(1,1,2,-110),hist(1,1,3,-120), ...
    hist(1,1,4,-124),hist(1,1,5,-130)];
calls=[call(1,1,3),call(1,1,5)];
populate(t.TestData.directory,h,calls,[-115 -126]);
reference=struct('status','TIME_LIMIT','objval',890,'lower',880,'api_key','secret-test-key'); %#ok<NASGU>
save(fullfile(t.TestData.parent,'full-reference.mat'),'reference');
s=suc.stagnation_online_report(t.TestData.directory); r=s.runs(2); b=r.blocks(1);
verifyEqual(t,r.attribution_status,'MATCHED'); verifyEqual(t,r.reference_difference,10);
verifyEqual(t,b.llm_direct_improvement,9);
verifyEqual(t,b.call_iteration_sa_improvement,7);
verifyEqual(t,b.later_noncall_sa_improvement,4);
verifyEqual(t,b.sa_after_first_call_improvement,6);
verifyEqual(t,r.attribution_coverage,'COMPLETED_HISTORY_ONLY');
verifyTrue(t,r.may_have_unrecorded_tail);
verifyFalse(t,contains(fileread(fullfile(t.TestData.directory,'online-analysis.json')),'secret-test-key'));
verifyFalse(t,isfile(fullfile(t.TestData.directory,'online-summary.json')));
end

function testNoCrossBlockOrOuterSubtraction(t)
h=[hist(1,1,1,-100),hist(1,1,2,-110),hist(1,1,3,-120), ...
    hist(1,1,4,-124),hist(2,1,3,-507),hist(2,1,4,-510)];
populate(t.TestData.directory,h,[call(1,1,3),call(2,1,3)],[-115 -500]);
s=suc.stagnation_online_report(t.TestData.directory); r=s.runs(2);
verifyEqual(t,r.attribution_status,'PARTIAL'); verifyEqual(t,numel(r.blocks),2);
verifyEqual(t,r.blocks(1).llm_direct_improvement,5);
verifyEqual(t,r.blocks(2).llm_direct_improvement,7);
verifyTrue(t,isnan(r.blocks(2).call_iteration_sa_improvement));
verifyEqual(t,r.blocks(2).sa_after_first_call_improvement,3);
end

function testUnmatchedCallCannotBecomeZeroImprovement(t)
populate(t.TestData.directory,[hist(1,1,2,-100),hist(1,1,3,-110)],call(1,1,3),-105);
path=fullfile(t.TestData.directory,'online-calls','call-ledger.json');
write_json(path,struct('started_utc','2026-10-07T01:00:00.000Z','directory','missing','status','OK'));
s=suc.stagnation_online_report(t.TestData.directory); r=s.runs(2);
verifyEqual(t,r.attribution_status,'UNKNOWN');
verifyEqual(t,r.attribution_reason,'CALL_COUNT_MISMATCH');
verifyTrue(t,isnan(r.blocks(1).llm_direct_improvement));
end

function testControlNeedsNoLedgerAndDIsSelected(t)
write_json(fullfile(t.TestData.parent,'summary.json'),struct('selected_mode','D'));
folder=fullfile(t.TestData.directory,'seed-107'); mkdir(folder);
result=struct('status','TIME_LIMIT','objective',900,'lower_bound',800, ...
    'relative_gap',1/9,'elapsed_seconds',1800,'llm_calls',0,'llm_history',[],'history',[]); %#ok<NASGU>
save(fullfile(folder,'A-online.mat'),'result');
s=suc.stagnation_online_report(t.TestData.directory);
verifyEqual(t,s.modes,'AD'); verifyEqual(t,s.runs(1).attribution_status,'NOT_APPLICABLE');
end

function testStopsValidationPairAndSameScreeningState(t)
populate(t.TestData.directory,[hist(1,1,2,-100),hist(1,1,3,-110)],call(1,1,3),-105);
folder=fullfile(t.TestData.directory,'seed-107');
loaded=load(fullfile(folder,'C-online.mat'),'result'); result=loaded.result;
result.status='INNER_NOT_CONVERGED'; result.elapsed_seconds=300;
result.validation=struct('feasible',true,'max_residual_mw',1e-10,'cost_residual',1e-8);
save(fullfile(folder,'C-online.mat'),'result');
result.objective=1000; result.llm_calls=0; result.llm_history=[]; %#ok<NASGU>
save(fullfile(folder,'A-online.mat'),'result');
for seed=[7 17 27]
    directory=fullfile(t.TestData.parent,sprintf('seed-%d',seed)); mkdir(directory);
    write_json(fullfile(directory,'capture-status.json'), ...
        struct('status','STAGNATION_CAPTURED','block',1,'inner',3,'upper',-58356));
end
s=suc.stagnation_online_report(t.TestData.directory);
verifyEqual(t,s.time_limit_seconds,1800); verifyEqual(t,s.max_inner,200);
verifyEqual(t,s.runs(2).last_outer,1); verifyEqual(t,s.runs(2).last_block,1);
verifyEqual(t,s.runs(2).last_inner,3); verifyEqual(t,s.runs(2).elapsed_seconds,300);
verifyEqual(t,s.runs(2).status,'INNER_NOT_CONVERGED');
verifyFalse(t,s.runs(2).may_have_unrecorded_tail);
verifyEqual(t,s.runs(2).validation_status,'VALIDATED');
verifyEqual(t,s.pairs(1).status,'VALIDATED'); verifyEqual(t,s.pairs(1).fusion_improvement,100);
verifyEqual(t,s.pairs(1).fusion_improvement_fraction,0.1,'AbsTol',1e-12);
verifyTrue(t,s.screening_context.same_block_iteration_and_cost);
end

function populate(directory,h,calls,costs)
folder=fullfile(directory,'seed-107'); mkdir(folder);
result=struct('status','TIME_LIMIT','objective',900,'lower_bound',800,'relative_gap',1/9, ...
    'elapsed_seconds',1800,'llm_calls',numel(calls),'llm_history',calls,'history',h); %#ok<NASGU>
save(fullfile(folder,'C-online.mat'),'result');
mkdir(fullfile(directory,'tasks'));
write_json(fullfile(directory,'tasks','online-107-C.done.json'), ...
    struct('started_utc','2026-10-07T00:00:00.0000000Z','ended_utc','2026-10-07T00:30:00.0000000Z'));
root=fullfile(directory,'online-calls'); mkdir(root); ledger=[];
for k=1:numel(costs)
    attempt=fullfile(root,sprintf('call-%d',k)); mkdir(attempt);
    write_json(fullfile(attempt,'request.json'),struct('input',jsonencode(struct('current_cost',costs(k)))));
    ledger=[ledger struct('directory',attempt,'started_utc', ...
        sprintf('2026-10-07T00:%02d:00.000Z',k),'status','OK')]; %#ok<AGROW>
end
ledger=[ledger struct('directory','other-run','started_utc','2026-10-07T01:00:00.000Z','status','OK')];
write_json(fullfile(root,'call-ledger.json'),ledger);
end

function h=hist(outer,block,inner,upper)
h=struct('outer',outer,'block',block,'inner',inner,'upper',upper,'lower',upper-10);
end

function c=call(outer,block,inner)
c=struct('outer',outer,'block',block,'inner',inner,'status','OK');
end

function write_json(path,data)
fid=fopen(path,'w','n','UTF-8'); cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(data));
end
