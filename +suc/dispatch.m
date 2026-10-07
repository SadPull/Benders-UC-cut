function r = dispatch(b,x,seconds)
%DISPATCH Convex recourse QP; elastic LP supplies infeasibility cuts.
started = tic; X = b.binary; Y = b.continuous;
r = struct('feasible',false,'status','DISCRETE_INFEASIBLE','cost',Inf,'point',[], ...
    'cut',[],'recourse',Inf);
assert(numel(x)==numel(X) && all(isfinite(x)) && all(x==0 | x==1), ...
    'suc:Candidate','Dispatch requires an exact binary vector.');
hard = ~any(b.A(:,Y),2);
v = b.A(hard,X)*x-b.rhs(hard); e = b.eq(hard); v(e) = abs(v(e));
if any(v > 1e-7), return; end
rows = find(~hard); B = b.A(rows,X);
g = struct('A',b.A(rows,Y),'rhs',b.rhs(rows)-B*x,'eq',b.eq(rows), ...
    'obj',b.obj(Y),'q',b.q(Y),'lb',b.lb(Y),'ub',b.ub(Y),'binary',[]);
raw = suc.optimize(g,seconds);
r.status = raw.status;
if strcmp(raw.status,'OPTIMAL')
    lambda = raw.lambda;
    beta = B'*lambda;
    r.cut = struct('beta',beta,'alpha',raw.lower-beta'*x,'optimality',true);
    r.point = zeros(numel(b.obj),1); r.point(X) = x; r.point(Y) = raw.x;
    r.recourse = raw.objval; r.cost = raw.objval+b.obj(X)'*x+b.constant;
    r.feasible = true;
elseif ismember(raw.status,{'INFEASIBLE','INF_OR_UNBD'}) && toc(started) < seconds
    % Separate nonnegative slack for each inequality; equalities get two sides.
    E = find(g.eq); C = [g.A;-g.A(E,:)]; D = [B;-B(E,:)];
    h = [b.rhs(rows);-b.rhs(rows(E))]; nr = numel(h); ny = numel(Y);
    elastic = struct('A',[C -speye(nr)],'rhs',h-D*x,'eq',false(nr,1), ...
        'obj',[zeros(ny,1);ones(nr,1)],'q',zeros(ny+nr,1), ...
        'lb',[g.lb;zeros(nr,1)],'ub',[g.ub;Inf(nr,1)],'binary',[]);
    raw = suc.optimize(elastic,seconds-toc(started)); r.status = raw.status;
    if ~strcmp(raw.status,'OPTIMAL'), return; end
    % 0 <= lambda <= 1 makes the slack part of the dual exactly feasible.
    lambda = min(1,max(0,raw.lambda));
    c = C'*lambda; z = g.lb; z(c < 0) = g.ub(c < 0);
    alpha = -h'*lambda + c'*z - 1e-7;
    r.cut = struct('beta',D'*lambda,'alpha',alpha,'optimality',false);
    r.recourse = raw.objval; r.status = 'INFEASIBLE';
end
end
