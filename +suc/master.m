function [m,lower] = master(b,cuts,seconds)
%MASTER Exact MILP for bound certification only; no candidate output channel.
X = b.binary; Y = b.continuous; nx = numel(X);
hard = ~any(b.A(:,Y),2);
A = b.A(hard,X); rhs = b.rhs(hard); eq = b.eq(hard);
kind = repmat({'hard'},numel(rhs),1);
map = zeros(numel(b.obj),1); map(X) = 1:nx;
and_pairs = zeros(0,3);
% Exact capacity: min(startup envelope, next-shutdown envelope).
for t = 1:numel(b.times)
    capacity = sparse(1,size(A,2));
    for i = 1:b.d.n_units
        u = map(b.u(i,t)); v = map(b.v(i,t));
        sr = b.d.pmax(i)-b.d.pstart(i); dr = b.d.pmax(i)-b.d.pshut(i);
        capacity(u) = capacity(u)+b.d.pmax(i); capacity(v) = capacity(v)-sr;
        if t < numel(b.times)
            w = map(b.w(i,t+1)); capacity(w) = capacity(w)-dr;
            if min(sr,dr) > 0
                z = size(A,2)+1; A(:,z) = 0; capacity(z) = min(sr,dr);
                and_pairs(end+1,:) = [z v w]; %#ok<AGROW>
                add([z v],[1 -1],0,false,'hard');
                add([z w],[1 -1],0,false,'hard');
                add([v w z],[1 1 -1],1,false,'hard');
            end
        end
    end
    required = (1+b.d.reserve_fraction)*max(arrayfun(@(s)s.load(b.times(t)),b.scenarios));
    [~,indices,values] = find(-capacity);
    add(indices,values,-required,false,'capacity');
end
nb = size(A,2); theta = nb+1; A(:,theta) = 0;
qlo = sum(min(b.obj(Y).*b.lb(Y),b.obj(Y).*b.ub(Y)));
add([1:nx theta],[b.obj(X)' -1],-qlo,false,'optimality');
for k = 1:numel(cuts)
    c = cuts(k);
    if c.optimality
        add([1:nx theta],[(c.beta+b.obj(X))' -1],-c.alpha,false,'optimality');
    else
        add(1:nx,c.beta',-c.alpha,false,'feasibility');
    end
end
lo = sum(min(b.obj.*b.lb,b.obj.*b.ub));
hi = sum(max(b.obj.*b.lb,b.obj.*b.ub)+b.q.*b.ub.^2);
m = struct('A',A,'rhs',rhs,'eq',eq,'obj',[zeros(nb,1);1], ...
    'q',zeros(theta,1),'lb',[zeros(nb,1);lo],'ub',[ones(nb,1);hi], ...
    'binary',(1:nb)','nx',nx,'and_pairs',and_pairs,'kind',{kind},'scale',b.scale);
lower = -Inf;
if nargin >= 3 && seconds > 0
    r = suc.optimize(m,seconds,true);
    if strcmp(r.status,'INFEASIBLE'), error('suc:Master','Certified master is infeasible.'); end
    lower = r.lower+b.constant-1e-5;
end
    function add(j,v,r,e,label)
        row = sparse(1,size(A,2)); row(j) = v;
        A(end+1,:) = row; rhs(end+1,1) = r; eq(end+1,1) = e; kind{end+1,1} = label;
    end
end
