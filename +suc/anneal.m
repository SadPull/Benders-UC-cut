function [candidates,info] = anneal(q,o,seconds,read_plan)
%ANNEAL Local single-bit Metropolis SA; QUBO energy is never a bound.
if nargin<4, read_plan=struct(); end
started = tic; scale=max(1,full(max(abs(q.Q),[],'all'))); Q=q.Q/scale;
n = size(Q,1); diagonal = full(diag(Q));
planned=isfield(read_plan,'second_start') && ~isempty(read_plan.second_start);
factor=1; parent='';
if planned
    validateattributes(read_plan.second_start,{'numeric'}, ...
        {'size',[n 1],'real','finite'},'','second_start');
    assert(all(read_plan.second_start==0 | read_plan.second_start==1), ...
        'suc:AnnealStart','Second read requires binary QUBO bits.');
    if isfield(read_plan,'second_temperature_factor'), factor=read_plan.second_temperature_factor; end
    validateattributes(factor,{'numeric'},{'scalar','positive','finite'});
    if isfield(read_plan,'parent_id'), parent=read_plan.parent_id; end
end
% Cool from constraint flip scale to objective resolution, as in the source.
flip_scale=max([q.energy_unit/scale;full(2*sum(abs(Q),2))+abs(diagonal)]);
cutoff=o.sa_cutoff*q.energy_unit/scale;
samples = q.starts; energies = zeros(1,size(samples,2));
for k=1:size(samples,2), energies(k)=samples(:,k)'*Q*samples(:,k); end
sources=struct('kind',{},'read',{},'seed_index',{},'parent_id',{});
for k=1:size(samples,2)
    sources(k)=struct('kind','seed','read',0,'seed_index',k,'parent_id',sprintf('seed:%d',k));
end
reads=struct('read',{},'start_bits',{},'output_bits',{},'start_binary',{}, ...
    'output_binary',{},'seed_index',{},'parent_id',{},'temperature_factor',{});
flips=0;
for read = 1:o.sa_reads
    if toc(started) >= seconds, break; end
    seed_index=0; parent_id=''; temperature_factor=1;
    if read==1 && ~isempty(q.starts)
        x=q.starts(:,1); seed_index=1; parent_id='seed:1';
    elseif read==2 && planned
        x=read_plan.second_start; parent_id=parent; temperature_factor=factor;
        match=find(all(q.starts==x,1),1);
        if ~isempty(match), seed_index=match; end
    else, x=double(rand(n,1)>0.5); end
    start=x;
    field = Q*x; energy=x'*field; best=x; best_energy=energy;
    temperature=o.sa_temperature*flip_scale*temperature_factor;
    while temperature > cutoff && toc(started) < seconds
        for sweep = 1:o.sa_sweeps
            for j = randperm(n)
                d=1-2*x(j); delta=2*d*field(j)+diagonal(j);
                if delta <= 0 || rand < exp(-delta/temperature)
                    x(j)=1-x(j); field=field+d*Q(:,j); energy=energy+delta; flips=flips+1;
                    if energy < best_energy, best=x; best_energy=energy; end
                end
            end
            if toc(started) >= seconds, break; end
        end
        temperature=temperature*o.sa_cooling;
    end
    samples(:,end+1)=best; energies(end+1)=best'*Q*best; %#ok<AGROW>
    sources(end+1)=struct('kind','sa','read',read,'seed_index',seed_index,'parent_id',parent_id); %#ok<AGROW>
    reads(end+1)=struct('read',read,'start_bits',start,'output_bits',best, ...
        'start_binary',decode(start),'output_binary',decode(best),'seed_index',seed_index, ...
        'parent_id',parent_id,'temperature_factor',temperature_factor); %#ok<AGROW>
end
[energies,order]=sort(energies); samples=samples(:,order);
sources=sources(order);
x = repmat(q.fixed,1,size(samples,2)); x(q.free,:)=samples(1:numel(q.free),:);
candidates=unique(x(1:q.nx,:)','rows','stable')';
candidate_sources=cell(1,size(candidates,2));
for k=1:size(candidates,2)
    candidate_sources{k}=sources(all(x(1:q.nx,:)==candidates(:,k),1));
end
info=struct('bits',n,'accepted_flips',flips,'best_energy',min([Inf energies])*scale+q.offset, ...
    'seconds',toc(started),'candidates',size(candidates,2), ...
    'read_history',reads,'candidate_sources',{candidate_sources});
    function binary=decode(bits)
        binary=q.fixed; binary(q.free)=bits(1:numel(q.free)); binary=binary(1:q.nx);
    end
end
