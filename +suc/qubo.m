function q = qubo(m,o,seeds,upper)
%QUBO Binary total-cost epigraph with bounded slack and conservative cuts.
nb = numel(m.binary); fixed = nan(nb,1);
hard = find(strcmp(m.kind,'hard'));
% Propagate exact consequences of hard rows before introducing slack bits.
changed = true;
while changed
    changed = false;
    for k = hard'
        [~,js,vs] = find(m.A(k,1:nb));
        js=js(:)'; vs=vs(:)';
        c = -m.rhs(k); known = ~isnan(fixed(js));
        c = c + vs(known)*fixed(js(known)); js = js(~known); vs = vs(~known);
        for j = 1:numel(js)
            rest = vs; rest(j) = [];
            lo = c+sum(min(0,rest)); hi = c+sum(max(0,rest));
            allowed = [lo lo+vs(j)] <= 1e-9;
            if m.eq(k), allowed = allowed & [hi hi+vs(j)] >= -1e-9; end
            assert(any(allowed),'suc:Qubo','Inconsistent hard constraints.');
            if nnz(allowed)==1
                fixed(js(j)) = find(allowed)-1; changed = true;
                break
            end
        end
    end
end
free = find(isnan(fixed)); known = find(~isnan(fixed));
lower = floor(m.lb(end)/o.theta_step)*o.theta_step;
upper = ceil(min(upper,m.ub(end))/o.theta_step)*o.theta_step;
assert(isfinite(upper) && upper >= lower,'suc:Qubo','Invalid epigraph range.');
weights = o.theta_step*bits((upper-lower)/o.theta_step);
theta_ids = numel(free)+(1:numel(weights));
nv = numel(free)+numel(weights);
Q = sparse(nv,nv); offset = lower/m.scale;
Q = Q + sparse(theta_ids,theta_ids,weights/m.scale,nv,nv);
slacks = struct('ids',{},'weights',{},'a',{},'constant',{});
for k = 1:size(m.A,1)
    a = [full(m.A(k,free)) full(m.A(k,end))*weights];
    c = full(m.A(k,known))*fixed(known)-m.rhs(k)+m.A(k,end)*lower;
    label = m.kind{k};
    if strcmp(label,'hard') && ~m.eq(k) && nnz(a)<=2
        % Two-bit implications/exclusivity have exact quadratic penalties;
        % their slack can be eliminated, as in the original compact QUBO.
        ids=find(a); values=a(ids); e00=max(0,c)^2;
        offset=offset+o.penalty*e00;
        if ~isempty(ids)
            e10=max(0,c+values(1))^2;
            Q(ids(1),ids(1))=Q(ids(1),ids(1))+o.penalty*(e10-e00);
            if numel(ids)==2
                e01=max(0,c+values(2))^2; e11=max(0,c+sum(values))^2;
                Q(ids(2),ids(2))=Q(ids(2),ids(2))+o.penalty*(e01-e00);
                mixed=o.penalty*(e11-e10-e01+e00)/2;
                Q(ids(1),ids(2))=Q(ids(1),ids(2))+mixed;
                Q(ids(2),ids(1))=Q(ids(2),ids(1))+mixed;
            end
        end
        continue
    end
    switch label
        case 'hard'
            a = round(a); c = round(c); scale = 1;
        case 'capacity'
            % Preserve the exact decimal capacity lattice (up to six places).
            raw = a*1e6;
            assert(all(abs(raw-round(raw))<1e-5) && abs(c*1e6-round(c*1e6))<1e-4, ...
                'suc:Qubo','Capacity data supports at most six decimal places.');
            ai = round(raw); divisor = 0;
            for value = abs(ai(ai~=0)), divisor = gcd(divisor,value); end
            if divisor==0, divisor=1; end
            a = ai/divisor; c = ceil(round(c*1e6)/divisor);
            scale = divisor/1e6/o.reserve_scale;
        otherwise
            if strcmp(label,'optimality'), step=o.theta_step; norm=m.scale;
            else, step=1e-3; norm=o.reserve_scale; end
            % floor for every coefficient and constant is a lower relaxation
            % on binary points. Exact unrounded cuts remain in certification.
            a = floor(a/step); c = floor(c/step); scale = step/norm;
    end
    low = c+sum(min(0,a)); high = c+sum(max(0,a));
    if ~m.eq(k) && high <= 0, continue; end
    if ~m.eq(k)
        sw = bits(max(0,-low)); ids = size(Q,1)+(1:numel(sw));
        slacks(end+1) = struct('ids',ids,'weights',sw,'a',a,'constant',c); %#ok<AGROW>
    else
        sw=[]; ids=[];
    end
    total = size(Q,1)+numel(sw);
    if total > size(Q,1), Q(total,total)=0; end
    idx = [1:numel(a) ids]; coeff = [a sw]*scale; constant = c*scale;
    row = sparse(1,idx,coeff,1,total);
    Q = Q+o.penalty*(row'*row+spdiags(2*constant*row',0,total,total));
    offset = offset+o.penalty*constant^2;
end
starts = zeros(size(Q,1),size(seeds,2));
for k = 1:size(seeds,2)
    x = zeros(nb,1); x(1:m.nx) = seeds(:,k);
    for j = 1:size(m.and_pairs,1)
        pair = m.and_pairs(j,:); x(pair(1)) = x(pair(2))*x(pair(3));
    end
    x(known) = fixed(known);
    rows = m.A(:,end) < 0;
    theta = max([lower; m.A(rows,1:nb)*x-m.rhs(rows)]);
    starts(1:numel(free),k) = x(free);
    starts(theta_ids,k) = encode(ceil((min(upper,theta)-lower)/o.theta_step),weights/o.theta_step);
    for j = 1:numel(slacks)
        z = slacks(j); value = -z.constant-z.a*starts(1:numel(z.a),k);
        starts(z.ids,k) = encode(round(value),z.weights);
    end
end
q = struct('Q',Q,'offset',offset,'fixed',fixed,'free',free,'nx',m.nx, ...
    'starts',starts,'theta_ids',theta_ids,'theta_weights',weights,'theta_lower',lower, ...
    'energy_unit',o.theta_step/m.scale);
end

function w = bits(maximum)
maximum = round(maximum);
assert(maximum >= 0 && maximum <= flintmax,'suc:Qubo','Integer encoding exceeds exact double range.');
if maximum==0, w=[]; return; end
n = ceil(log2(maximum+1)); w = [2.^(0:n-2) maximum-(2^(n-1)-1)];
end

function x = encode(value,w)
x = zeros(numel(w),1); value = min(sum(w),max(0,value));
for j = numel(w):-1:1
    if value >= w(j), x(j)=1; value=value-w(j); end
end
end
