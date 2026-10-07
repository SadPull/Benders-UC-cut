function tests=test_search_checkpoint
tests=functiontests(localfunctions);
end

function setupOnce(t)
global SEARCH_CHECKPOINT_PLANS;
SEARCH_CHECKPOINT_PLANS={};
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
folder=t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
mkdir(fullfile(folder.Folder,'+suc'));
write_stub(folder.Folder,'master',[
    "function [m,bound]=master(varargin)"
    "global SEARCH_CHECKPOINT_DELAYS;"
    "SEARCH_CHECKPOINT_DELAYS.master_calls=SEARCH_CHECKPOINT_DELAYS.master_calls+1;"
    "if SEARCH_CHECKPOINT_DELAYS.master>0, pause(SEARCH_CHECKPOINT_DELAYS.master); end"
    "m=struct('lb',[0;0]); bound=SEARCH_CHECKPOINT_DELAYS.master_bound;"
    "end"]);
write_stub(folder.Folder,'dispatch',[
    "function r=dispatch(b,x,seconds)"
    "global SEARCH_CHECKPOINT_DELAYS;"
    "SEARCH_CHECKPOINT_DELAYS.dispatch_calls=SEARCH_CHECKPOINT_DELAYS.dispatch_calls+1;"
    "if SEARCH_CHECKPOINT_DELAYS.dispatch>0, pause(SEARCH_CHECKPOINT_DELAYS.dispatch); end"
    "cost=100+sum(x);"
    "r=struct('feasible',true,'status','OPTIMAL','cost',cost,'point',[x;100], ..."
    " 'cut',struct('alpha',cost,'beta',zeros(size(x)),'optimality',true));"
    "end"]);
write_stub(folder.Folder,'qubo',[
    "function q=qubo(m,o,seeds,upper)"
    "q=struct('starts',seeds);"
    "end"]);
write_stub(folder.Folder,'anneal',[
    "function [x,info]=anneal(q,o,seconds,plan)"
    "global SEARCH_CHECKPOINT_PLANS;"
    "if nargin<4, plan=[]; end"
    "SEARCH_CHECKPOINT_PLANS{end+1}=plan;"
    "rand; x=q.starts;"
    "info=struct('bits',size(x,1),'accepted_flips',0,'best_energy',0,'seconds',0,'candidates',size(x,2));"
    "end"]);
write_stub(folder.Folder,'llm',[
    "function [x,info]=llm(b,best,lower,o,seconds,feedback)"
    "global SEARCH_CHECKPOINT_FEEDBACK; SEARCH_CHECKPOINT_FEEDBACK=feedback;"
    "x=[0 1 1;0 0 1;0 0 0];"
    "info=struct('status','OK','accepted',3);"
    "end"]);
t.applyFixture(matlab.unittest.fixtures.PathFixture(folder.Folder));
end

function setup(~)
global SEARCH_CHECKPOINT_DELAYS;
SEARCH_CHECKPOINT_DELAYS=struct('master',0,'dispatch',0,'master_bound',0, ...
    'master_calls',0,'dispatch_calls',0);
end

function teardown(~)
clear global SEARCH_CHECKPOINT_DELAYS;
end

function testResumeMatchesUninterrupted(t)
[b,o]=fixture(); rng(13);
[whole,whole_history,whole_state]=suc.search_block(b,ones(3,1),o,tic,0,1,1);
rng(13);
[first,~,checkpoint]=suc.search_block(b,ones(3,1),o,tic,0,1,1,[],struct('mode','capture'));
verifyTrue(t,first.captured);
verifyEqual(t,checkpoint.phase,'after_sa'); verifyEqual(t,checkpoint.iteration,3);
verifyEqual(t,numel(checkpoint.history),2);
[resumed,resumed_history,resumed_state]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,struct('mode','A'));
verifyEqual(t,resumed.upper,whole.upper); verifyEqual(t,resumed.lower,whole.lower);
verifyEqual(t,resumed_state.cache,whole_state.cache);
verifyEqual(t,resumed_state.cuts,whole_state.cuts);
verifyEqual(t,resumed_history,whole_history);
verifyEqual(t,resumed_state.rng,whole_state.rng);
end

function testChargePreventsLateCandidateAcceptance(t)
[b,o]=fixture();
[~,~,checkpoint]=suc.search_block(b,ones(3,1),o,tic,0,1,1,[],struct('mode','capture'));
intervention=struct('mode','C','proposals',zeros(3,1),'charged_seconds',o.time_limit);
[out,~,state]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,intervention);
verifyEqual(t,out.upper,checkpoint.upper);
verifyEqual(t,state.cache,checkpoint.cache);
verifyGreaterThanOrEqual(t,out.elapsed_seconds,o.time_limit);
end

function testCandidateReplayAndFiveBatchHeat(t)
global SEARCH_CHECKPOINT_PLANS;
[b,o]=fixture(); o.max_inner=10;
[~,~,checkpoint]=suc.search_block(b,ones(3,1),o,tic,0,1,1,[],struct('mode','capture'));
proposals=[zeros(3,1) [1;0;0] [1;1;0]];
SEARCH_CHECKPOINT_PLANS={};
[c,~,cs]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,struct('mode','C','proposals',proposals));
verifyTrue(t,all(cellfun(@isempty,SEARCH_CHECKPOINT_PLANS)));
SEARCH_CHECKPOINT_PLANS={};
[d,~,ds]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,struct('mode','D','proposals',proposals));
verifyEqual(t,c.upper,100); verifyEqual(t,d.upper,c.upper);
verifyEqual(t,cs.cache,ds.cache);
verifyEqual(t,sum(~cellfun(@isempty,SEARCH_CHECKPOINT_PLANS)),5);
verifyEqual(t,SEARCH_CHECKPOINT_PLANS{1}.second_start,[1;0;0]);
verifyEqual(t,SEARCH_CHECKPOINT_PLANS{2}.second_start,[1;1;0]);
verifyEqual(t,SEARCH_CHECKPOINT_PLANS{1}.second_temperature_factor,1);
verifyEqual(t,sum(strcmp({d.evaluation_events.source},'LLM') & [d.evaluation_events.newbest]),1);
SEARCH_CHECKPOINT_PLANS={};
suc.search_block(b,[],o,tic,0,1,1,checkpoint,struct('mode','B'));
verifyEqual(t,sum(~cellfun(@isempty,SEARCH_CHECKPOINT_PLANS)),5);
verifyEqual(t,SEARCH_CHECKPOINT_PLANS{1}.second_start,ones(3,1));
verifyEqual(t,SEARCH_CHECKPOINT_PLANS{1}.second_temperature_factor,2);
end

function testDispatchReturningAfterDeadlineCannotCommit(t)
global SEARCH_CHECKPOINT_DELAYS;
[b,o]=fixture();
[~,~,checkpoint]=suc.search_block(b,ones(3,1),o,tic,0,1,1,[],struct('mode','capture'));
o.time_limit=0.25;
SEARCH_CHECKPOINT_DELAYS.dispatch=0.35;
SEARCH_CHECKPOINT_DELAYS.dispatch_calls=0;
SEARCH_CHECKPOINT_DELAYS.master_calls=0;
intervention=struct('mode','C','proposals',zeros(3,1));
[out,~,state]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,intervention);
% The improving QP was actually started, then returned after the deadline.
verifyEqual(t,SEARCH_CHECKPOINT_DELAYS.dispatch_calls,1);
verifyEqual(t,SEARCH_CHECKPOINT_DELAYS.master_calls,0);
verifyGreaterThanOrEqual(t,out.elapsed_seconds,o.time_limit);
verifyEqual(t,out.upper,checkpoint.upper);
verifyEqual(t,state.cache,checkpoint.cache);
verifyEqual(t,state.cuts,checkpoint.cuts);
verifyEqual(t,state.best,checkpoint.best);
verifyEmpty(t,out.evaluation_events);
end

function testCertificationReturningAfterDeadlineCannotCommit(t)
global SEARCH_CHECKPOINT_DELAYS;
[b,o]=fixture();
[~,~,checkpoint]=suc.search_block(b,ones(3,1),o,tic,0,1,1,[],struct('mode','capture'));
o.time_limit=0.25;
SEARCH_CHECKPOINT_DELAYS.master=0.35;
SEARCH_CHECKPOINT_DELAYS.master_bound=checkpoint.upper-1;
SEARCH_CHECKPOINT_DELAYS.master_calls=0;
SEARCH_CHECKPOINT_DELAYS.dispatch_calls=0;
[out,~,state]=suc.search_block(b,[],o,tic,0,1,1,checkpoint,struct('mode','A'));
% A tighter certificate was produced, but only after its allowed time.
verifyEqual(t,SEARCH_CHECKPOINT_DELAYS.master_calls,1);
verifyEqual(t,SEARCH_CHECKPOINT_DELAYS.dispatch_calls,0);
verifyGreaterThanOrEqual(t,SEARCH_CHECKPOINT_DELAYS.master_bound,checkpoint.lower+1);
verifyGreaterThanOrEqual(t,out.elapsed_seconds,o.time_limit);
verifyEqual(t,out.lower,checkpoint.lower);
verifyEqual(t,state.lower,checkpoint.lower);
verifyEqual(t,out.upper,checkpoint.upper);
end

function testOnlineFusionAndMinimumRemaining(t)
global SEARCH_CHECKPOINT_PLANS SEARCH_CHECKPOINT_FEEDBACK;
[b,o]=fixture(); o.max_inner=10; o.llm_enabled=true; o.llm_max_calls=1;
o.fusion_mode='D'; SEARCH_CHECKPOINT_PLANS={}; SEARCH_CHECKPOINT_FEEDBACK=[];
[out,~,state]=suc.search_block(b,ones(3,1),o,tic,0,1,1);
verifyEqual(t,out.llm_calls,1); verifyEqual(t,out.upper,100);
verifyEqual(t,sum(~cellfun(@isempty,SEARCH_CHECKPOINT_PLANS)),5);
verifyLessThanOrEqual(t,numel(SEARCH_CHECKPOINT_FEEDBACK.recent),12);
verifyLessThanOrEqual(t,numel(SEARCH_CHECKPOINT_FEEDBACK.cuts),8);
verifyEqual(t,state.llm_history.inner,3);
o.fusion_mode='C'; SEARCH_CHECKPOINT_PLANS={};
out=suc.search_block(b,ones(3,1),o,tic,0,1,1);
verifyEqual(t,out.llm_calls,1);
verifyTrue(t,all(cellfun(@isempty,SEARCH_CHECKPOINT_PLANS)));
o.llm_min_remaining=o.time_limit+1; SEARCH_CHECKPOINT_FEEDBACK=[];
out=suc.search_block(b,ones(3,1),o,tic,0,1,1);
verifyEqual(t,out.llm_calls,0); verifyEmpty(t,SEARCH_CHECKPOINT_FEEDBACK);
end

function [b,o]=fixture()
b=struct('obj',ones(4,1),'q',zeros(4,1),'lb',zeros(4,1),'ub',[ones(3,1);200], ...
    'constant',0,'binary',(1:3)','A',[0 0 0 1],'rhs',100,'eq',true);
o=struct('max_inner',7,'time_limit',30,'certify_seconds',1,'tolerance',1e-4, ...
    'llm_enabled',false,'llm_patience',3,'llm_max_calls',0);
end

function write_stub(root,name,lines)
fid=fopen(fullfile(root,'+suc',[name '.m']),'w');
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',lines);
end
