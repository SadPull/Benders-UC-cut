function [blocks,cross] = partition(m,hours,lambda)
%PARTITION Keep cross-block rows in the Lagrangian, without resetting time.
if nargin < 3, lambda = zeros(size(m.rhs)); end
owner = ceil(m.time/hours); B = max(owner); nr = numel(m.rhs);
[row,col] = find(m.A);
first = accumarray(row,owner(col),[nr 1],@min);
last = accumarray(row,owner(col),[nr 1],@max);
cross = find(first ~= last);
assert(all(lambda(~m.eq) >= -1e-10),'suc:Price','Inequality multipliers must be nonnegative.');
lambda(~m.eq) = max(0,lambda(~m.eq));
costs = m.obj + m.A(cross,:)'*lambda(cross);
blocks = cell(B,1);
for j = 1:B
    cols = find(owner == j); rows = find(first == j & last == j);
    b = struct('A',m.A(rows,cols),'rhs',m.rhs(rows),'eq',m.eq(rows), ...
        'obj',costs(cols),'q',m.q(cols),'lb',m.lb(cols),'ub',m.ub(cols), ...
        'cols',cols,'rows',rows,'time',m.time(cols),'times',unique(m.time(cols))', ...
        'constant',-m.rhs(cross(first(cross)==j))'*lambda(cross(first(cross)==j)));
    b.binary = find(ismember(cols,m.binary));
    b.continuous = find(~ismember(cols,m.binary));
    b.scale = max(1,sum(abs(m.obj(cols)).*b.ub + b.q.*b.ub.^2));
    b.crossA = m.A(cross,cols);
    b.crossrhs = m.rhs(cross).*(first(cross)==j);
    map = zeros(numel(m.obj),1); map(cols) = 1:numel(cols);
    b.u = reshape(map(m.ix.u(:,b.times)),m.d.n_units,[]);
    b.v = reshape(map(m.ix.v(:,b.times)),m.d.n_units,[]);
    b.w = reshape(map(m.ix.w(:,b.times)),m.d.n_units,[]);
    b.original_obj = m.obj(cols); b.d = m.d; b.scenarios = m.scenarios;
    blocks{j} = b;
end
end
