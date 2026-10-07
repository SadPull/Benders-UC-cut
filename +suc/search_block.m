function [out,history,state] = search_block(b,seeds,o,started,used_calls,outer,index,state,intervention)
%SEARCH_BLOCK Block Benders search with a resumable post-SA checkpoint.
if nargin<8, state=[]; end
if nargin<9, intervention=struct(); end
if ~isfield(intervention,'mode'), intervention.mode='none'; end
if ~isfield(intervention,'charged_seconds'), intervention.charged_seconds=0; end
if ~isfield(o,'fusion_mode'), o.fusion_mode='C'; end
if ~isfield(o,'llm_min_remaining'), o.llm_min_remaining=0; end
mode=intervention.mode;
assert(ismember(mode,{'none','capture','A','B','C','D'}),'suc:Intervention','Unknown intervention.');
validateattributes(intervention.charged_seconds,{'numeric'},{'scalar','nonnegative','finite'});
event_start=0;
if isempty(state)
    seeds=unique(seeds','rows','stable')';
    state=struct('cuts',struct('beta',{},'alpha',{},'optimality',{}), ...
        'cache',[],'pool',{{}},'best',[],'lower',sum(min(b.obj.*b.lb,b.obj.*b.ub))+b.constant, ...
        'upper',Inf,'seeds',seeds,'iteration',1,'stalled',0,'calls',0,'llm_history',[], ...
        'history',[],'phase','before_sa','sampling',empty_sampling(),'events',[], ...
        'intervention_applied',false,'replay_counted',false,'intervention_batches',0,'llm_points',{{}}, ...
        'llm_costs',[],'reference_binary',[],'rng',rng);
    for k=1:size(seeds,2), evaluate(seeds(:,k),'initial',[]); end
else
    rng(state.rng);
    event_start=numel(state.events);
end
if ismember(mode,{'C','D'}) && ~state.replay_counted
    state.calls=state.calls+1; state.replay_counted=true;
end
captured=false;
while state.iteration<=o.max_inner && remaining()>0 && ~strcmp(state.phase,'done')
    if strcmp(state.phase,'before_sa')
        before=state.upper;
        [master,bound]=suc.master(b,state.cuts,min(o.certify_seconds,remaining()));
        if remaining()<=0, break; end
        state.lower=max(state.lower,bound);
        state.sampling=empty_sampling();
        if ~closed(state.lower,state.upper,o.tolerance)
            seed_pool=state.seeds;
            if ~isempty(state.best), seed_pool=[state.best.point(b.binary) seed_pool]; end
            read_plan=[];
            use_llm_start=strcmp(mode,'D') || (strcmp(mode,'none') && strcmp(o.fusion_mode,'D'));
            special=(strcmp(mode,'B') || use_llm_start) && ...
                state.intervention_applied && state.intervention_batches<5;
            selected=[];
            if special && use_llm_start
                eligible=find(cellfun(@(p)~isequal(p,state.best.point(b.binary)),state.llm_points));
                if ~isempty(eligible)
                    selected=eligible(1+mod(state.intervention_batches,numel(eligible)));
                    seed_pool=[seed_pool state.llm_points{selected}];
                end
            end
            master.lb(end)=max(master.lb(end),state.lower-b.constant);
            q=suc.qubo(master,o,seed_pool,state.upper-b.constant);
            if special && strcmp(mode,'B') && ~isempty(state.best)
                read_plan=struct('second_start',q.starts(:,1),'second_temperature_factor',2, ...
                    'parent_id',fingerprint(state.best.point(b.binary)));
            elseif special && ~isempty(selected)
                read_plan=struct('second_start',q.starts(:,end),'second_temperature_factor',1, ...
                    'parent_id',fingerprint(state.llm_points{selected}));
            end
            if remaining()>0
                if isempty(read_plan)
                    [proposals,state.sampling]=suc.anneal(q,o,remaining());
                else
                    [proposals,state.sampling]=suc.anneal(q,o,remaining(),read_plan);
                end
                if state.intervention_applied, state.intervention_batches=state.intervention_batches+1; end
                for k=1:size(proposals,2)
                    origins=[];
                    if isfield(state.sampling,'candidate_sources'), origins=state.sampling.candidate_sources{k}; end
                    evaluate(proposals(:,k),'SA',origins);
                end
            end
        end
        if remaining()<=0, break; end
        if isfinite(before) && before-state.upper<=5e-4*max(1,abs(before))
            state.stalled=state.stalled+1;
        else
            state.stalled=0;
        end
        state.phase='after_sa';
        if strcmp(mode,'capture') && ~isempty(state.best) && state.stalled>=3 && ...
                ~closed(state.lower,state.upper,o.tolerance)
            state.reference_binary=state.best.point(b.binary);
            captured=true; break;
        end
    end
    if strcmp(state.phase,'after_sa')
        if ismember(mode,{'A','B','C','D'}) && ~state.intervention_applied
            state.intervention_applied=true;
            if ismember(mode,{'C','D'})
                assert(isfield(intervention,'proposals'),'suc:Intervention','Missing shared candidates.');
                for k=1:size(intervention.proposals,2)
                    x=intervention.proposals(:,k); r=evaluate(x,'LLM',[]);
                    if ~isempty(r) && r.feasible && remaining()>0 && ...
                            ~any(cellfun(@(p)isequal(p,x),state.llm_points))
                        state.llm_points{end+1}=x; state.llm_costs(end+1)=r.cost;
                    end
                end
                [state.llm_costs,order]=sort(state.llm_costs);
                state.llm_points=state.llm_points(order);
                state.stalled=0;
            end
        elseif strcmp(mode,'none') && o.llm_enabled && ~isempty(state.best) && ...
                state.stalled>=o.llm_patience && used_calls+state.calls<o.llm_max_calls && ...
                remaining()>0 && remaining()>=o.llm_min_remaining && ...
                ~closed(state.lower,state.upper,o.tolerance)
            feedback=struct('recent',state.events(max(1,end-11):end), ...
                'cuts',state.cuts(max(1,end-7):end));
            [proposals,info]=suc.llm(b,state.best,state.lower,o,remaining(),feedback);
            state.calls=state.calls+1; state.stalled=0;
            info.outer=outer; info.block=index; info.inner=state.iteration;
            state.llm_history=[state.llm_history info];
            if strcmp(o.fusion_mode,'D')
                state.llm_points={}; state.llm_costs=[];
                state.intervention_applied=true; state.intervention_batches=0;
            end
            for k=1:size(proposals,2)
                x=proposals(:,k); r=evaluate(x,'LLM',[]);
                if strcmp(o.fusion_mode,'D') && ~isempty(r) && r.feasible && remaining()>0 && ...
                        ~any(cellfun(@(p)isequal(p,x),state.llm_points))
                    state.llm_points{end+1}=x; state.llm_costs(end+1)=r.cost;
                end
            end
            [state.llm_costs,order]=sort(state.llm_costs);
            state.llm_points=state.llm_points(order);
        end
        if remaining()>0
            [~,bound]=suc.master(b,state.cuts,min(o.certify_seconds,remaining()));
            if remaining()>0, state.lower=max(state.lower,bound); end
        end
        state.history=[state.history struct('outer',outer,'block',index,'inner',state.iteration, ...
            'lower',state.lower,'upper',state.upper,'optimality_cuts',sum([state.cuts.optimality]), ...
            'feasibility_cuts',sum(~[state.cuts.optimality]),'sampling',state.sampling)];
        state.iteration=state.iteration+1; state.phase='before_sa';
        if closed(state.lower,state.upper,o.tolerance), state.phase='done'; end
    end
end
state.rng=rng;
history=state.history;
out=struct('lower',state.lower,'upper',state.upper,'best',state.best,'pool',{state.pool}, ...
    'seeds',state.seeds,'llm_calls',state.calls,'llm_history',state.llm_history, ...
    'captured',captured,'events',state.events,'intervention_batches',state.intervention_batches, ...
    'evaluation_events',state.events(event_start+1:end), ...
    'charged_seconds',intervention.charged_seconds,'elapsed_seconds',elapsed());
if ~isempty(state.best), out.seeds=state.best.point(b.binary); end
    function seconds=remaining()
        seconds=o.time_limit-elapsed();
    end
    function seconds=elapsed()
        seconds=toc(started)+intervention.charged_seconds;
    end
    function r=evaluate(x,source,origins)
        r=[];
        if remaining()<=0, return; end
        if ~isempty(state.cache) && any(all(state.cache==x,1))
            for p=1:numel(state.pool)
                point=state.pool{p};
                if isequal(point(b.binary),x)
                    r=struct('feasible',true,'cost',b.obj'*point+sum(b.q.*point.^2)+b.constant);
                    break;
                end
            end
            record(x,source,origins,'DUPLICATE',NaN,state.upper,state.upper,0); return;
        end
        began=elapsed(); before_cost=state.upper;
        r=suc.dispatch(b,x,remaining());
        if remaining()<=0, r=[]; return; end
        if r.feasible
            residual=b.A*r.point-b.rhs; residual(b.eq)=abs(residual(b.eq));
            assert(max(residual)<=1e-5,'suc:Dispatch','Local dispatch validation failed.');
        elseif ~ismember(r.status,{'INFEASIBLE','DISCRETE_INFEASIBLE'})
            error('suc:Dispatch','Local dispatch failed: %s',r.status);
        end
        if remaining()<=0, r=[]; return; end
        state.cache(:,end+1)=x;
        if ~isempty(r.cut), state.cuts(end+1)=r.cut; end
        if r.feasible
            state.pool{end+1}=r.point;
            if r.cost<state.upper, state.best=r; state.upper=r.cost; end
        end
        record(x,source,origins,r.status,r.cost,before_cost,state.upper,elapsed()-began);
    end
    function record(x,source,origins,status,cost,before_cost,after_cost,seconds)
        distance=NaN;
        if ~isempty(state.reference_binary), distance=sum(x~=state.reference_binary); end
        parent={};
        if isstruct(origins) && isfield(origins,'parent_id'), parent={origins.parent_id}; end
        event=struct('inner',state.iteration,'source',source,'origins',{origins}, ...
            'parent',{parent},'newbest',after_cost<before_cost, ...
            'candidate_id',fingerprint(x),'status',status,'cost',cost, ...
            'before',before_cost,'after',after_cost,'improvement',before_cost-after_cost, ...
            'distance_to_checkpoint',distance,'seconds',seconds,'completed_seconds',elapsed(), ...
            'elapsed',elapsed());
        state.events=[state.events event];
    end
end

function info=empty_sampling()
info=struct('bits',0,'accepted_flips',0,'best_energy',NaN,'seconds',0,'candidates',0);
end

function yes=closed(lower,upper,tol)
yes=isfinite(lower) && isfinite(upper) && upper-lower>=-1e-5 && ...
    upper-lower<=1e-5+tol*max(1,abs(upper));
end

function id=fingerprint(x)
id=char(double('0')+x(:)');
end
