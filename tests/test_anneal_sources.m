function tests = test_anneal_sources
tests=functiontests(localfunctions);
end

function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));
t.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testOptionalPlanPreservesRandomStream(t)
[q,o]=fixture(); state=rng; cleanup=onCleanup(@()rng(state)); %#ok<NASGU>
rng(91); [a,ia]=suc.anneal(q,o,10); after=rng;
rng(91); [b,ib]=suc.anneal(q,o,10,struct());
verifyEqual(t,a,b); verifyEqual(t,rng,after);
verifyEqual(t,ia.read_history,ib.read_history);
verifyEqual(t,ia.candidate_sources,ib.candidate_sources);
end

function testSecondReadPlanAndFirstRead(t)
[q,o]=fixture();
plan=struct('second_start',q.starts(:,2),'second_temperature_factor',2,'parent_id','llm:010');
[~,info]=suc.anneal(q,o,10,plan);
verifyEqual(t,numel(info.read_history),o.sa_reads);
verifyEqual(t,info.read_history(1).start_bits,q.starts(:,1));
verifyEqual(t,info.read_history(1).temperature_factor,1);
verifyEqual(t,info.read_history(2).start_bits,plan.second_start);
verifyEqual(t,info.read_history(2).temperature_factor,2);
verifyEqual(t,info.read_history(2).parent_id,'llm:010');
verifyEqual(t,info.read_history(2).seed_index,2);
verifyEqual(t,info.read_history(3).temperature_factor,1);
verifyEqual(t,info.read_history(3).seed_index,0);
end

function testSeedReturnIsNotAnnealingDiscovery(t)
[q,o]=fixture(); q.starts=[q.starts q.starts(:,1)];
[x,info]=suc.anneal(q,o,0);
verifyEmpty(t,info.read_history);
verifyEqual(t,size(x,2),2);
for k=1:size(x,2)
    origins=info.candidate_sources{k};
    verifyTrue(t,all(strcmp({origins.kind},'seed')));
    verifyTrue(t,all([origins.read]==0));
end
k=find(all(x==[0;1;0],1));
verifyEqual(t,[info.candidate_sources{k}.seed_index],[1 3]);
end

function testDuplicateOutputRetainsSeedAndReadSources(t)
% Zero energy makes every read return its initial state, even after flips.
[q,o]=fixture(); q.Q=sparse(2,2); o.sa_reads=2;
plan=struct('second_start',q.starts(:,1),'parent_id','llm-parent');
[x,info]=suc.anneal(q,o,10,plan);
k=find(all(x==[0;1;0],1)); origins=info.candidate_sources{k};
verifyEqual(t,sum(strcmp({origins.kind},'seed')),1);
verifyEqual(t,sort([origins(strcmp({origins.kind},'sa')).read]),[1 2]);
verifyEqual(t,info.read_history(2).start_binary,[0;1;0]);
verifyEqual(t,info.read_history(2).output_binary,[0;1;0]);
verifyTrue(t,any(strcmp({origins.parent_id},'llm-parent')));
end

function testRejectNonbinaryPlannedStart(t)
[q,o]=fixture();
verifyError(t,@()suc.anneal(q,o,1,struct('second_start',[0;0.5])), ...
    'suc:AnnealStart');
end

function [q,o]=fixture()
q=struct('Q',sparse([2 -1;-1 3]),'offset',0,'fixed',[NaN;1;NaN], ...
    'free',[1;3],'nx',3,'starts',[0 1;0 1],'energy_unit',1);
o=struct('sa_reads',3,'sa_sweeps',1,'sa_temperature',1, ...
    'sa_cooling',0.5,'sa_cutoff',0.1);
end
