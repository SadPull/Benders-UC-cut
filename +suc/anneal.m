function [candidates,info] = anneal(q,o,seconds)
%ANNEAL Local single-bit Metropolis SA; QUBO energy is never a bound.
started = tic; scale=max(1,full(max(abs(q.Q),[],'all'))); Q=q.Q/scale;
n = size(Q,1); diagonal = full(diag(Q));
% Cool from constraint flip scale to objective resolution, as in the source.
flip_scale=max([q.energy_unit/scale;full(2*sum(abs(Q),2))+abs(diagonal)]);
cutoff=o.sa_cutoff*q.energy_unit/scale;
samples = q.starts; energies = zeros(1,size(samples,2));
for k=1:size(samples,2), energies(k)=samples(:,k)'*Q*samples(:,k); end
flips=0;
for read = 1:o.sa_reads
    if toc(started) >= seconds, break; end
    if read==1 && ~isempty(q.starts), x=q.starts(:,1);
    else, x=double(rand(n,1)>0.5); end
    field = Q*x; energy=x'*field; best=x; best_energy=energy;
    temperature=o.sa_temperature*flip_scale;
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
end
[energies,order]=sort(energies); samples=samples(:,order);
x = repmat(q.fixed,1,size(samples,2)); x(q.free,:)=samples(1:numel(q.free),:);
candidates=unique(x(1:q.nx,:)','rows','stable')';
info=struct('bits',n,'accepted_flips',flips,'best_energy',min([Inf energies])*scale+q.offset, ...
    'seconds',toc(started),'candidates',size(candidates,2));
end
