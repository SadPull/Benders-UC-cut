function tests = test_stagnation_report
tests=functiontests(localfunctions);
end

function setup(t)
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
fixture=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
t.TestData.directory=fixture.Folder;
end

function testMissingArtifactsRemainPending(t)
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.status,'PENDING'); verifyFalse(t,s.expand);
verifyTrue(t,all(strcmp({s.records.status},'PENDING')));
verifyTrue(t,isfile(fullfile(t.TestData.directory,'summary.json')));
verifyTrue(t,isfile(fullfile(t.TestData.directory,'curves.csv')));
end

function testDQualifiedDoesNotAccidentallySelectC(t)
populate(t.TestData.directory,999.6,999.2);
s=suc.stagnation_report(t.TestData.directory);
verifyFalse(t,s.qualified_C); verifyTrue(t,s.qualified_D);
verifyFalse(t,s.expand); verifyEmpty(t,s.selected_mode);
verifyEqual(t,s.decision,'D_QUALIFIED_BUT_SELECTION_THRESHOLD_NOT_MET');
end

function testSelectDAndCountOnlyCompletedNewBatches(t)
populate(t.TestData.directory,1000,999);
s=suc.stagnation_report(t.TestData.directory);
verifyTrue(t,s.expand); verifyEqual(t,s.selected_mode,'D');
verifyEqual(t,s.records(1).common_batches,2);
b=s.records(1).branches(4);
verifyEqual(t,b.common_batch_upper,999);
verifyEqual(t,b.llm_direct_improvement,0.6,'AbsTol',1e-12);
verifyEqual(t,b.sa_improvement,0.4,'AbsTol',1e-12);
verifyEqual(t,b.first_breakthrough_seconds,21);
raw=fileread(fullfile(t.TestData.directory,'summary.json'));
verifyFalse(t,contains(raw,'secret-test-key'));
csv=fileread(fullfile(t.TestData.directory,'curves.csv'));
verifyFalse(t,contains(csv,'1201'));
end

function testSelectSimplerQualifiedC(t)
populate(t.TestData.directory,999,998.9);
s=suc.stagnation_report(t.TestData.directory);
verifyTrue(t,s.qualified_C); verifyTrue(t,s.qualified_D);
verifyEqual(t,s.selected_mode,'C');
end

function testStrictThresholdAndNoTrigger(t)
populate(t.TestData.directory,999.5,999.5);
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.control_wins_C,0); verifyFalse(t,s.expand);
result=struct('status','TIME_LIMIT'); %#ok<NASGU>
save(fullfile(t.TestData.directory,'seed-27','capture.mat'),'result');
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.records(3).status,'NO_TRIGGER');
verifyEmpty(t,s.records(3).branches);
end

function testPendingFullCheckPreventsExpansion(t)
populate(t.TestData.directory,1000,999);
path=fullfile(t.TestData.directory,'seed-7','A.mat');
loaded=load(path,'branch'); branch=loaded.branch;
branch.full_check=struct('status','PENDING','feasible',false,'objective',Inf); %#ok<NASGU>
save(path,'branch');
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.status,'PENDING'); verifyFalse(t,s.expand);
verifyEqual(t,s.records(1).branches(1).status,'PENDING');
end

function testOptionalReferencesAndSupportedRead(t)
populate(t.TestData.directory,1000,999);
reference=struct('status','TIME_LIMIT','lower',980,'objval',998,'api_key','secret-test-key'); %#ok<NASGU>
save(fullfile(t.TestData.directory,'seed-7','block-reference.mat'),'reference');
reference=struct('status','OPTIMAL','lower',4990,'objval',4990); %#ok<NASGU>
save(fullfile(t.TestData.directory,'full-reference.mat'),'reference');
path=fullfile(t.TestData.directory,'seed-7','D.mat');
loaded=load(path,'branch'); branch=loaded.branch;
for k=1:numel(branch.events)
    branch.events(k).newbest=true;
    branch.events(k).origins=struct('kind','sa','read',2,'parent_id','llm-parent');
end
save(path,'branch');
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.records(1).block_reference.status,'TIME_LIMIT');
verifyEqual(t,s.records(1).block_reference.lower,980);
verifyEqual(t,s.records(1).block_reference.objective,998);
verifyFalse(t,s.records(2).block_reference.available);
verifyEqual(t,s.full_reference.objective,4990);
b=s.records(1).branches(4);
verifyEqual(t,b.supported_second_read_improvement,0.4,'AbsTol',1e-12);
verifyEqual(t,b.supported_second_read_newbests,1);
verifyFalse(t,contains(fileread(fullfile(t.TestData.directory,'summary.json')),'secret-test-key'));
% A seed-only origin and a read without a parent are not supported ancestry.
branch.events(2).origins(1).kind='seed';
branch.events(2).origins(2)=struct('kind','sa','read',2,'parent_id','');
save(path,'branch');
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.records(1).branches(4).supported_second_read_newbests,0);
end

function testIndependentFullArtifactOverridesEmbeddedCheck(t)
populate(t.TestData.directory,1000,999);
folder=fullfile(t.TestData.directory,'seed-7');
loaded=load(fullfile(folder,'D.mat'),'branch'); branch=loaded.branch;
branch.full_check=struct('status','PENDING','feasible',false,'objective',Inf); %#ok<NASGU>
save(fullfile(folder,'D.mat'),'branch');
full_check=struct('status','OPTIMAL','feasible',true,'objective',4998); %#ok<NASGU>
save(fullfile(folder,'D-full.mat'),'full_check');
s=suc.stagnation_report(t.TestData.directory);
verifyEqual(t,s.status,'COMPLETE'); verifyTrue(t,s.expand);
verifyEqual(t,s.records(1).branches(4).full_objective,4998);
verifyEqual(t,s.records(1).branches(4).full_improvement,2);
end

function populate(directory,c_cost,d_cost)
for seed=[7 17 27]
    folder=fullfile(directory,sprintf('seed-%d',seed)); mkdir(folder);
    result=struct('checkpoint',struct('state',struct('iteration',3), ...
        'incumbent',struct('objective',5000)), 'api_key','secret-test-key'); %#ok<NASGU>
    save(fullfile(folder,'capture.mat'),'result');
    shared=struct('status','OK','charged_seconds',20,'api_key','secret-test-key'); %#ok<NASGU>
    save(fullfile(folder,'response.mat'),'shared');
    modes='ABCD'; costs=[1000 1000 c_cost d_cost];
    for j=1:4
        history=struct('inner',{},'upper',{},'sampling',{});
        for inner=2:5
            history(end+1)=struct('inner',inner,'upper',costs(j), ...
                'sampling',struct('seconds',1)); %#ok<AGROW>
        end
        events=[];
        if j>=3
            events=[event('LLM',21,1000,999.4),event('SA',30,999.4,999), ...
                event('SA',1201,999,900)];
        end
        branch=struct('mode',modes(j),'seed',seed,'baseline_cost',1000, ...
            'out',struct('upper',costs(j),'lower',990,'elapsed_seconds',100, ...
                'intervention_batches',2),'history',history,'events',events, ...
            'full_check',struct('objective',4999,'feasible',true,'status','OPTIMAL')); %#ok<NASGU>
        save(fullfile(folder,[modes(j) '.mat']),'branch');
    end
end
end

function e=event(source,seconds,before,after)
e=struct('source',source,'completed_seconds',seconds,'before',before,'after',after, ...
    'improvement',before-after);
end
