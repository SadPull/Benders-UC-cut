function result = solve(d,s,o)
%SOLVE Strict nested block Benders/SA and Lagrangian cutting-plane loop.
started=tic; m=suc.model(d,s); [full,~]=suc.partition(m,d.horizon); full=full{1};
result=struct('status','MAX_ITERATIONS','message','','objective',Inf,'lower_bound',-Inf, ...
    'relative_gap',Inf,'coordination_upper',Inf,'u',[],'v',[],'w',[], ...
    'scenarios',[],'validation',[],'history',[],'outer_history',[], ...
    'llm_calls',0,'llm_history',[],'elapsed_seconds',0);
incumbent=[]; stage='INITIALIZATION';
try
    relaxation=suc.optimize(m,remaining());
    assert(strcmp(relaxation.status,'OPTIMAL'),'suc:Initialization', ...
        'Continuous relaxation failed: %s',relaxation.status);
    result.lower_bound=relaxation.lower-1e-5; lambda=relaxation.lambda;
    initial={round(reshape(relaxation.x(m.ix.u),d.n_units,d.horizon)),ones(d.n_units,d.horizon)};
    for j=1:numel(initial)
        if remaining()<=0, break; end
        accept_full(suc.repair(d,s,initial{j}));
    end
    if isempty(incumbent)
        if remaining()<=0, result.status='TIME_LIMIT';
        else, result.status='INITIALIZATION_FAILED'; result.message='No validated initial schedule.'; end
    else
        [blocks,cross]=suc.partition(m,o.block_hours,lambda); B=numel(blocks);
        planes=cell(B,1); previous=cell(B,1);
        for outer=1:o.max_outer
            if remaining()<=0, result.status='TIME_LIMIT'; break; end
            [blocks,~]=suc.partition(m,o.block_hours,lambda);
            bounds=cellfun(@(b)sum(min(b.obj.*b.lb,b.obj.*b.ub))+b.constant,blocks);
            complete=true;
            for j=1:B
                if remaining()<=0, result.status='TIME_LIMIT'; complete=false; break; end
                stage=sprintf('BLOCK_%d_%d',outer,j); b=blocks{j};
                seeds=[incumbent.point(b.cols(b.binary)) previous{j}];
                [search,log]=search_block(b,seeds,o,started,result.llm_calls,outer,j);
                result.llm_calls=result.llm_calls+search.llm_calls;
                result.llm_history=[result.llm_history search.llm_history];
                result.history=[result.history log];
                bounds(j)=max(bounds(j),search.lower);
                result.lower_bound=max(result.lower_bound,sum(bounds));
                previous{j}=search.seeds;
                for k=1:numel(search.pool)
                    p=search.pool{k};
                    plane=struct('intercept',b.original_obj'*p+sum(b.q.*p.^2), ...
                        'slope',b.crossA*p-b.crossrhs);
                    planes{j}=[planes{j} plane];
                end
                if ~isempty(search.best) && remaining()>0
                    proposal=incumbent.u;
                    proposal(:,b.times)=search.best.point(b.u);
                    accept_full(suc.repair(d,s,proposal));
                end
                if o.verbose
                    fprintf('outer=%d block=%d/%d  local=[%.6f, %.6f]  full=%.6f  LB=%.6f\n', ...
                        outer,j,B,search.lower,search.upper,result.objective,result.lower_bound);
                end
                if remaining()<=0, result.status='TIME_LIMIT'; complete=false; break; end
                if ~closed(search.lower,search.upper,o.tolerance)
                    result.status='INNER_NOT_CONVERGED'; complete=false; break;
                end
            end
            if ~complete, break; end
            stage='COORDINATION';
            [prices,upper]=suc.coordinate(planes,m.eq(cross),result.objective,remaining());
            if remaining()<=0, result.status='TIME_LIMIT'; break; end
            result.coordination_upper=upper;
            result.outer_history=[result.outer_history struct('outer',outer, ...
                'lower',result.lower_bound,'coordination_upper',upper,'multipliers',prices)];
            if closed(result.lower_bound,upper,o.tolerance,result.lower_bound)
                result.status='DUAL_CONVERGED'; break;
            end
            lambda=zeros(size(m.rhs)); lambda(cross)=prices;
        end
    end
catch exception
    if remaining()<=0 || strcmp(exception.identifier,'suc:TimeLimit'), result.status='TIME_LIMIT';
    else, result.status=[stage '_FAILED']; end
    result.message=exception.message;
end
if isfinite(result.objective)
    result.validation=suc.validate(d,s,result);
    assert(result.lower_bound <= result.objective+1e-4,'suc:Bounds','Lower bound exceeds validated cost.');
    result.relative_gap=max(0,result.objective-result.lower_bound)/max(1,abs(result.objective));
end
result.elapsed_seconds=toc(started);
if o.verbose
    fprintf('%s: cost=%.6f, lower=%.6f, gap=%.4f%%, elapsed=%.2fs\n', ...
        result.status,result.objective,result.lower_bound,100*result.relative_gap,result.elapsed_seconds);
end
    function seconds=remaining()
        seconds=o.time_limit-toc(started);
    end
    function accept_full(x)
        if isempty(x) || remaining()<=0, return; end
        if ~isempty(incumbent) && isequal(x,incumbent.point(full.binary)), return; end
        r=suc.dispatch(full,x,remaining());
        if ~r.feasible || remaining()<=0, return; end
        z=unpack(m,r.point,d,s); z.validation=suc.validate(d,s,z);
        if z.objective < result.objective
            incumbent=z; incumbent.point=r.point;
            for f={'objective','u','v','w','scenarios','validation'}, result.(f{1})=z.(f{1}); end
        end
    end
end

function [out,history]=search_block(b,seeds,o,started,used_calls,outer,index)
cuts=struct('beta',{},'alpha',{},'optimality',{}); cache=[]; pool={}; best=[];
lower=sum(min(b.obj.*b.lb,b.obj.*b.ub))+b.constant; upper=Inf;
history=[]; stalled=0; calls=0; llm_history=[];
seeds=unique(seeds','rows','stable')';
for k=1:size(seeds,2), evaluate(seeds(:,k)); end
for iteration=1:o.max_inner
    if remaining()<=0, break; end
    before=upper;
    [master,bound]=suc.master(b,cuts,min(o.certify_seconds,remaining())); lower=max(lower,bound);
    sampling=struct('bits',0,'accepted_flips',0,'best_energy',NaN,'seconds',0,'candidates',0);
    if ~closed(lower,upper,o.tolerance) && remaining()>0
        seed_pool=seeds;
        if ~isempty(best), seed_pool=[best.point(b.binary) seed_pool]; end
        master.lb(end)=max(master.lb(end),lower-b.constant);
        q=suc.qubo(master,o,seed_pool,upper-b.constant);
        if remaining()>0
            [proposals,sampling]=suc.anneal(q,o,remaining());
            for k=1:size(proposals,2), evaluate(proposals(:,k)); end
        end
    end
    if isfinite(before) && before-upper <= 5e-4*max(1,abs(before)), stalled=stalled+1;
    else, stalled=0; end
    if o.llm_enabled && ~isempty(best) && stalled>=o.llm_patience && ...
            used_calls+calls<o.llm_max_calls && remaining()>0 && ~closed(lower,upper,o.tolerance)
        [proposals,info]=suc.llm(b,best,lower,o,remaining()); calls=calls+1; stalled=0;
        info.outer=outer; info.block=index; info.inner=iteration;
        llm_history=[llm_history info]; %#ok<AGROW>
        for k=1:size(proposals,2), evaluate(proposals(:,k)); end
    end
    if remaining()>0
        [~,bound]=suc.master(b,cuts,min(o.certify_seconds,remaining())); lower=max(lower,bound);
    end
    history=[history struct('outer',outer,'block',index,'inner',iteration,'lower',lower, ...
        'upper',upper,'optimality_cuts',sum([cuts.optimality]), ...
        'feasibility_cuts',sum(~[cuts.optimality]),'sampling',sampling)]; %#ok<AGROW>
    if closed(lower,upper,o.tolerance), break; end
end
out=struct('lower',lower,'upper',upper,'best',best,'pool',{pool}, ...
    'seeds',seeds,'llm_calls',calls,'llm_history',llm_history);
if ~isempty(best), out.seeds=best.point(b.binary); end
    function seconds=remaining()
        seconds=o.time_limit-toc(started);
    end
    function evaluate(x)
        if remaining()<=0 || (~isempty(cache) && any(all(cache==x,1))), return; end
        r=suc.dispatch(b,x,remaining());
        if remaining()<=0, return; end
        cache(:,end+1)=x;
        if ~isempty(r.cut), cuts(end+1)=r.cut; end
        if r.feasible
            residual=b.A*r.point-b.rhs; residual(b.eq)=abs(residual(b.eq));
            assert(max(residual)<=1e-5,'suc:Dispatch','Local dispatch validation failed.');
            pool{end+1}=r.point;
            if r.cost<upper, best=r; upper=r.cost; end
        elseif ~ismember(r.status,{'INFEASIBLE','DISCRETE_INFEASIBLE'})
            error('suc:Dispatch','Local dispatch failed: %s',r.status);
        end
    end
end

function yes=closed(lower,upper,tol,reference)
if nargin<4, reference=upper; end
yes=isfinite(lower) && isfinite(upper) && upper-lower>=-1e-5 && ...
    upper-lower<=1e-5+tol*max(1,abs(reference));
end

function z=unpack(m,point,d,s)
z=struct('objective',m.obj'*point+sum(m.q.*point.^2), ...
    'u',reshape(point(m.ix.u),d.n_units,d.horizon), ...
    'v',reshape(point(m.ix.v),d.n_units,d.horizon), ...
    'w',reshape(point(m.ix.w),d.n_units,d.horizon),'scenarios',[]);
for k=1:numel(s)
    p=reshape(point(m.ix.p(:,:,k)),d.n_units,d.horizon);
    r=reshape(point(m.ix.r(:,:,k)),d.n_units,d.horizon);
    z.scenarios=[z.scenarios struct('name',s(k).name,'probability',s(k).probability, ...
        'objective',sum(d.cost_a.*p.^2+d.cost_b.*p,'all'),'p',p,'r',r)];
end
end
