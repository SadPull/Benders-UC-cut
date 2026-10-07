function tests = test_core
tests=functiontests(localfunctions);
end

function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testDataAndOptions(t)
root=fileparts(fileparts(mfilename('fullpath')));
[d,s]=suc.load_data(fullfile(root,'data','units.json'),fullfile(root,'data','stochastic.json'));
verifyEqual(t,d.n_units,32); verifyEqual(t,d.horizon,24);
verifyEqual(t,sum(d.pmax),3405); verifyEqual(t,max([s.load])*(1+d.reserve_fraction),3142.125,'AbsTol',1e-8);
verifyEqual(t,sum([s.probability]),1);
verifyError(t,@()run_suc(struct('unknown',1)),'suc:Options');
end

function testPartitionAndPricedIdentity(t)
[d,s]=tiny(2,5); d.min_up(:)=4; d.min_down(:)=3;
m=suc.model(d,s); rng(1); point=rand(numel(m.obj),1);
for hours=[1 2 3 5]
    prices=rand(numel(m.rhs),1); prices(m.eq)=-prices(m.eq);
    [blocks,cross]=suc.partition(m,hours,prices);
    rows=cross; columns=[]; cost=0; constants=zeros(numel(cross),1);
    for j=1:numel(blocks)
        b=blocks{j}; rows=[rows;b.rows]; columns=[columns;b.cols]; %#ok<AGROW>
        p=point(b.cols); cost=cost+b.obj'*p+sum(b.q.*p.^2)+b.constant;
        constants=constants+b.crossrhs;
    end
    verifyEqual(t,sort(rows),(1:numel(m.rhs))');
    verifyEqual(t,sort(columns),(1:numel(m.obj))');
    verifyEqual(t,constants,m.rhs(cross));
    original=m.obj'*point+sum(m.q.*point.^2)+prices(cross)'*(m.A(cross,:)*point-m.rhs(cross));
    verifyEqual(t,cost,original,'AbsTol',1e-8);
    verifyEqual(t,blocks{end}.times(end),5);
end
end

function testCutsOnAllSmallCommitments(t)
[d,s]=tiny(2,3); s.load=[30 0 0]; d.base_load=s.load;
m=suc.model(d,s); blocks=suc.partition(m,3); b=blocks{1};
records={}; cuts=struct('beta',{},'alpha',{},'optimality',{}); points=[];
for code=0:63
    u=reshape(double(bitget(code,1:6)),2,3); x=pack(d,u);
    r=suc.dispatch(b,x,10);
    if ~isempty(r.cut)
        cuts(end+1)=r.cut; records{end+1}=r; points(:,end+1)=x; %#ok<AGROW>
        if r.feasible
            verifyEqual(t,r.cut.alpha+r.cut.beta'*x,r.recourse,'AbsTol',2e-5);
        else
            verifyGreaterThan(t,r.cut.alpha+r.cut.beta'*x,1e-6);
        end
    end
end
verifyTrue(t,any([cuts.optimality])); verifyTrue(t,any(~[cuts.optimality]));
cost=Inf;
for j=1:numel(records)
    if ~records{j}.feasible, continue; end
    cost=min(cost,records{j}.cost);
    for k=1:numel(cuts)
        value=cuts(k).alpha+cuts(k).beta'*points(:,j);
        if cuts(k).optimality, limit=records{j}.recourse; else, limit=0; end
        verifyLessThanOrEqual(t,value,limit+1e-5);
    end
end
[master,bound]=suc.master(b,cuts,10);
verifyEqual(t,bound,cost,'AbsTol',2e-4);
verifyFalse(t,isfield(master,'x')); % certification exposes no MIP candidate
reference=suc.optimize(m,10,true);
verifyEqual(t,reference.objval,cost,'AbsTol',2e-5);
end

function testStartupShutdownCapacity(t)
[d,s]=tiny(1,3); s.load=[35 0 0]; d.base_load=s.load;
m=suc.model(d,s); blocks=suc.partition(m,3); b=blocks{1}; x=pack(d,[1 0 0]);
r=suc.dispatch(b,x,10); verifyTrue(t,r.feasible);
s.load(1)=35.1; m=suc.model(d,s); blocks=suc.partition(m,3);
r=suc.dispatch(blocks{1},x,10); verifyFalse(t,r.feasible);
verifyGreaterThan(t,r.cut.alpha+r.cut.beta'*x,0);
end

function testQuboSlackAndEnergy(t)
% Exhaust all encodings: two decisions, a cost register and two slacks.
m=struct('A',sparse([1 1 0;1 2 -1]),'rhs',[1;0],'eq',[false;false], ...
    'obj',[0;0;1],'q',zeros(3,1),'lb',[0;0;0],'ub',[1;1;4], ...
    'binary',[1;2],'nx',2,'and_pairs',zeros(0,3),'kind',{{'hard';'optimality'}},'scale',10);
o=settings(); q=suc.qubo(m,o,[0 1 0;0 0 1],4); n=size(q.Q,1);
verifyLessThanOrEqual(t,n,12);
minimum=Inf(4,5);
for code=0:2^n-1
    bits=double(bitget(code,1:n))';
    x=q.fixed; x(q.free)=bits(1:numel(q.free)); theta=q.theta_lower+q.theta_weights*bits(q.theta_ids);
    index=1+x(1)+2*x(2); energy=bits'*q.Q*bits+q.offset;
    minimum(index,theta+1)=min(minimum(index,theta+1),energy);
end
for code=0:3
    x=double(bitget(code,1:2))';
    for theta=0:4
        if all(m.A*[x;theta]<=m.rhs)
            verifyEqual(t,minimum(code+1,theta+1),theta/m.scale,'AbsTol',1e-8);
        else
            verifyGreaterThan(t,minimum(code+1,theta+1),theta/m.scale+1);
        end
    end
end
for j=1:size(q.starts,2)
    bits=q.starts(:,j); theta=q.theta_lower+q.theta_weights*bits(q.theta_ids);
    verifyEqual(t,bits'*q.Q*bits+q.offset,theta/m.scale,'AbsTol',1e-8);
end
end

function testConservativeQuantizationAndCapacityLattice(t)
m=struct('A',sparse([-0.15 -0.25 0;0.27 -0.31 -1]),'rhs',[-0.2;-0.18], ...
    'eq',[false;false],'obj',[0;0;1],'q',zeros(3,1),'lb',[0;0;0],'ub',[1;1;1], ...
    'binary',[1;2],'nx',2,'and_pairs',zeros(0,3), ...
    'kind',{{'capacity';'optimality'}},'scale',1);
o=settings(); o.theta_step=0.1; q=suc.qubo(m,o,[0 1;1 1],1);
for j=1:size(q.starts,2)
    bits=q.starts(:,j); theta=q.theta_lower+q.theta_weights*bits(q.theta_ids);
    verifyEqual(t,bits'*q.Q*bits+q.offset,theta,'AbsTol',2e-7);
end
end

function testCoordinationFreeEqualityPrice(t)
planes={[struct('intercept',-2,'slope',-1),struct('intercept',2,'slope',1)]};
[lambda,upper]=suc.coordinate(planes,true,10,10);
verifyEqual(t,lambda,-2,'AbsTol',1e-7); verifyEqual(t,upper,0,'AbsTol',1e-7);
end

function testLlmParsingOffline(t)
% Shadow only the HTTP transport; no external service or real key is used.
folder=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
good=struct('u',ones(2,3),'v',[ones(2,1) zeros(2,2)],'w',zeros(2,3));
bad=good; bad.u(1,1)=0.25;
content=jsonencode(struct('candidates',[good bad]));
file=fullfile(folder.Folder,'webwrite.m'); fid=fopen(file,'w');
fprintf(fid,['function r=webwrite(url,body,settings)\n' ...
    'folder=fileparts(mfilename(''fullpath''));\n' ...
    'save(fullfile(folder,''request.mat''),''url'',''body'',''settings'');\n' ...
    'data=load(fullfile(folder,''response.mat''));\n' ...
    'if isfield(data.response,''transport_error''), error(''test:HttpFailure'',''Offline failure.''); end\n' ...
    'r=jsondecode(jsonencode(data.response));\nend\n']);
fclose(fid);
t.applyFixture(matlab.unittest.fixtures.PathFixture(folder.Folder));
[d,s]=tiny(2,3); m=suc.model(d,s); blocks=suc.partition(m,3); b=blocks{1};
best=suc.dispatch(b,pack(d,ones(2,3)),10);
o=settings(); o.llm_model='offline-test'; o.llm_url='https://unused.invalid/v1/responses';
o.llm_api_key='offline-test-key'; o.llm_timeout=10;
part=struct('type','output_text','text',content);
message=struct('type','message','role','assistant','content',part);
reasoning=struct('type','reasoning','summary',[]);
split=message;
split.content=[struct('type','output_text','text',content(1:20)) ...
    struct('type','output_text','text',content(21:end))];
refusal=message; refusal.content=struct('type','refusal','refusal','Cannot comply.');
invalid=message; invalid.content.text='not JSON';
responses={struct('status','completed','output',message), ...
    struct('status','completed','output',{{reasoning split}}), ...
    struct('status','completed','output',{{reasoning refusal}}), ...
    struct('status','incomplete','output',message), ...
    struct('status','completed','output',invalid), ...
    struct('status','completed','output',[]), ...
    struct('transport_error',true)};
for k=1:numel(responses)
    response=responses{k}; save(fullfile(folder.Folder,'response.mat'),'response');
    [candidates,info]=suc.llm(b,best,0,o,4);
    if k<=2
        verifyEqual(t,info.status,'OK'); verifyEqual(t,info.accepted,1);
        verifyEqual(t,candidates,pack(d,ones(2,3)));
    else
        verifyEqual(t,info.status,'FAILED'); verifyEmpty(t,candidates);
        verifyEqual(t,info.accepted,0);
    end
end
request=load(fullfile(folder.Folder,'request.mat'));
verifyEqual(t,request.url,o.llm_url);
verifyEqual(t,request.body.model,o.llm_model);
verifyFalse(t,request.body.stream); verifyFalse(t,request.body.store);
verifyFalse(t,isfield(request.body,'messages'));
verifySubstring(t,request.body.instructions,'Return ONLY JSON');
context=jsondecode(request.body.input);
verifyEqual(t,context.unit_count,2); verifyEqual(t,context.u,ones(2,3));
verifyEqual(t,request.settings.HeaderFields,{'Authorization','Bearer offline-test-key'});
verifyEqual(t,request.settings.Timeout,4);
end

function testLlmConfigValidation(t)
% A local configuration is loaded even when run_suc receives no arguments.
folder=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
fid=fopen(fullfile(folder.Folder,'llm_config.m'),'w');
fprintf(fid,['function o=llm_config\n' ...
    'o=struct(''llm_enabled'',true,''llm_url'',''https://unused.invalid/v1/responses'', ...\n' ...
    '''llm_model'',''offline-test'',''llm_api_key'','''',''llm_timeout'',10, ...\n' ...
    '''llm_patience'',3,''llm_max_calls'',3);\nend\n']);
fclose(fid);
t.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(folder.Folder));
verifyError(t,@()run_suc(),'suc:Options');
% Explicit options still override the local file, including an empty key.
verifyError(t,@()run_suc(struct('llm_api_key','offline-test-key','llm_model','')),'suc:Options');
result=run_suc(struct('llm_enabled',false,'time_limit',1e-9,'verbose',false));
verifyEqual(t,result.status,'TIME_LIMIT');
end

function testNestedConvergenceAndIndependentValidation(t)
[d,s]=tiny(1,3); s.load(:)=30; d.base_load=s.load; o=settings();
for hours=[1 2 3 5]
    o.block_hours=hours; result=suc.solve(d,s,o);
    verifyEqual(t,result.status,'DUAL_CONVERGED',result.message);
    verifyEqual(t,result.objective,225,'AbsTol',1e-5);
    verifyLessThanOrEqual(t,result.lower_bound,result.objective+1e-5);
    report=suc.validate(d,s,result); verifyTrue(t,report.feasible);
    bad=result; bad.scenarios.p(1,1)=bad.scenarios.p(1,1)+1;
    verifyError(t,@()suc.validate(d,s,bad),'suc:Validation');
end
end

function testDeadline(t)
[d,s]=tiny(1,3); o=settings(); o.time_limit=1e-9;
r=suc.solve(d,s,o); verifyEqual(t,r.status,'TIME_LIMIT'); verifyFalse(t,isfinite(r.objective));
end

function testRts24SmokeAndSeedReproducibility(t)
o=struct('max_outer',1,'max_inner',2,'sa_reads',1,'sa_sweeps',1,'verbose',false,'time_limit',60, ...
    'llm_enabled',false);
prior=rng; a=run_suc(o); verifyEqual(t,rng,prior);
b=run_suc(o);
verifyTrue(t,isfinite(a.objective),a.message);
verifyTrue(t,ismember(a.status,{'INNER_NOT_CONVERGED','MAX_ITERATIONS','DUAL_CONVERGED'}),a.message);
verifyTrue(t,a.validation.feasible); verifyGreaterThan(t,numel(a.history),0);
verifyGreaterThan(t,a.history(1).sampling.bits,0);
verifyLessThanOrEqual(t,a.lower_bound,a.objective+1e-5);
verifyEqual(t,a.u,b.u); verifyEqual(t,a.objective,b.objective,'AbsTol',1e-6);
verifyEqual(t,a.history(1).sampling.best_energy,b.history(1).sampling.best_energy,'AbsTol',1e-7);
end

function [d,s]=tiny(n,T)
d=struct('n_units',n,'horizon',T,'pmax',55*ones(n,1),'pmin',10*ones(n,1), ...
    'pstart',45*ones(n,1),'pshut',35*ones(n,1),'min_up',ones(n,1),'min_down',ones(n,1), ...
    'ramp_up',135*ones(n,1),'ramp_down',135*ones(n,1),'initial_on',zeros(n,1), ...
    'initial_duration',ones(n,1),'initial_output',zeros(n,1),'base_load',30*ones(1,T), ...
    'cost_a',0.01*ones(n,1),'cost_b',2*ones(n,1),'cost_c',5*ones(n,1), ...
    'startup_cost',3*ones(n,1),'shutdown_cost',4*ones(n,1),'reserve_fraction',0);
s=struct('name','tiny','probability',1,'load',d.base_load);
end

function x=pack(d,u)
previous=[d.initial_on u(:,1:end-1)]; v=max(u-previous,0); w=max(previous-u,0);
x=[u(:);v(:);w(:)];
end

function o=settings()
o=struct('max_outer',10,'max_inner',5,'block_hours',3,'tolerance',1e-4, ...
    'time_limit',30,'certify_seconds',5,'seed',7,'verbose',false, ...
    'theta_step',1,'penalty',1e6,'reserve_scale',100,'sa_reads',2,'sa_sweeps',1, ...
    'sa_temperature',20,'sa_cooling',0.9,'sa_cutoff',0.1, ...
    'llm_enabled',false,'llm_patience',3,'llm_max_calls',0);
end
