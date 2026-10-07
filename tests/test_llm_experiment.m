function tests = test_llm_experiment
tests=functiontests(localfunctions);
end

function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
[d,s]=suc.load_data(fullfile(root,'data','units.json'),fullfile(root,'data','stochastic.json'));
m=suc.model(d,s); blocks=suc.partition(m,3); t.TestData.b=blocks{2};
end

function testRequestIsOfflineAndContainsPricedContext(t)
b=t.TestData.b; point=(1:numel(b.obj))'; point(b.u)=1;
best=struct('cost',123,'point',point,'cut',struct('alpha',2, ...
    'beta',zeros(numel(b.binary),1),'optimality',true));
feedback=struct('cuts',best.cut,'recent',struct('status','DISCRETE_INFEASIBLE','duplicate',false));
[body,info]=suc.llm_request(b,best,0,struct('llm_model','offline-test'),feedback);
c=jsondecode(body.input);
verifyEqual(t,c.hours(:),b.times(:));
verifyEqual(t,c.scenario_probability(:),[b.scenarios.probability]');
verifyEqual(t,c.scenario_reserve,b.d.reserve_fraction*c.scenario_loads,'AbsTol',1e-10);
verifyEqual(t,c.original_binary_cost,b.original_obj(b.binary));
verifyEqual(t,c.lagrangian_binary_price,b.obj(b.binary)-b.original_obj(b.binary));
verifyEqual(t,c.local_constraints.u,b.u);
N=b.d.n_units*b.d.horizon;
ids=reshape(1:N,b.d.n_units,b.d.horizon); ids=ids(:,b.times);
[~,p]=ismember(3*N+ids,b.cols);
[~,r]=ismember(3*N+numel(b.scenarios)*N+ids,b.cols);
verifyEqual(t,c.dispatch(1).p,point(p)); verifyEqual(t,c.dispatch(1).r,point(r));
a=c.local_constraints.triplets;
verifyEqual(t,sparse(a(:,1),a(:,2),a(:,3),size(b.A,1),size(b.A,2)),b.A);
verifyEqual(t,c.recent_candidates.status,'DISCRETE_INFEASIBLE');
verifyFalse(t,isfield(body,'headers')); verifyFalse(t,body.store);
verifyGreaterThanOrEqual(t,info.context_seconds,0);
end

function testSyntaxCountsAreNotFeasibilityCounts(t)
b=t.TestData.b;
good=struct('u',zeros(size(b.u)),'v',zeros(size(b.v)),'w',zeros(size(b.w)));
bad=good; bad.u(1)=0.5;
[x,info]=suc.llm_parse(response({good,good,bad}),b);
verifyEqual(t,info.status,'OK'); verifyEqual(t,info.format_valid,2);
verifyEqual(t,info.accepted,1); verifyTrue(t,isnan(info.feasible));
verifyEqual(t,x,zeros(numel(b.binary),1));
end

function testMalformedRowsDoNotDiscardGoodRows(t)
b=t.TestData.b;
good=struct('u',ones(size(b.u)),'v',zeros(size(b.v)),'w',zeros(size(b.w)));
[x,info]=suc.llm_parse(response({struct('u',0),good}),b);
verifyEqual(t,info.accepted,1); verifyEqual(t,x(1:numel(b.u)),ones(numel(b.u),1));
r=response({good}); r.status='incomplete';
[x,info]=suc.llm_parse(r,b); verifyEmpty(t,x); verifyEqual(t,info.status,'FAILED');
[x,info]=suc.llm_parse(response({struct('u',0),struct('u',0),struct('u',0),good}),b);
verifyEmpty(t,x); verifyEqual(t,info.accepted,0);
end

function testGlobalBudgetConsumesFailedTransport(t)
folder=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
fid=fopen(fullfile(folder.Folder,'webwrite.m'),'w');
fprintf(fid,['function response=webwrite(varargin)\n' ...
    'error(''test:OfflineTransport'',''offline test transport failure'');\nend\n']); fclose(fid);
t.applyFixture(matlab.unittest.fixtures.PathFixture(folder.Folder));
b=t.TestData.b; best=struct('cost',123,'point',b.lb,'cut',[]);
o=struct('llm_model','offline-test','llm_url','https://unused.invalid/v1/responses', ...
    'llm_api_key','offline-private-key','llm_timeout',1, ...
    'llm_budget_dir',fullfile(folder.Folder,'budget'),'llm_global_max_calls',1);
[x,info]=suc.llm(b,best,0,o,1);
verifyEmpty(t,x); verifyEqual(t,info.message,'test:OfflineTransport');
entries=jsondecode(fileread(fullfile(o.llm_budget_dir,'call-ledger.json'),'Encoding','UTF-8'));
verifyEqual(t,numel(entries),1); verifyEqual(t,entries.status,'FAILED');
request=fileread(fullfile(entries.directory,'request.json'),'Encoding','UTF-8');
verifyFalse(t,contains(request,o.llm_api_key));
verifyTrue(t,isfile(fullfile(entries.directory,'info.json')));
[x,info]=suc.llm(b,best,0,o,1);
verifyEmpty(t,x); verifyEqual(t,info.message,'suc:LlmBudget');
entries=jsondecode(fileread(fullfile(o.llm_budget_dir,'call-ledger.json'),'Encoding','UTF-8'));
verifyEqual(t,numel(entries),1);
end

function r=response(rows)
part=struct('type','output_text','text',jsonencode(struct('candidates',{rows})));
r=struct('status','completed','output',struct('type','message','content',part));
end
