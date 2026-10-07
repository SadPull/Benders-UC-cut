function r = optimize(m,seconds,integral)
%OPTIMIZE Gurobi interface with a recomputed, box-minimized QP dual bound.
if nargin < 3, integral = false; end
assert(seconds > 0,'suc:TimeLimit','No time remaining for optimization.');
g.A = m.A; g.rhs = m.rhs; g.sense = repmat('<',size(m.rhs)); g.sense(m.eq) = '=';
g.obj = m.obj; g.lb = m.lb; g.ub = m.ub; g.modelsense = 'min';
if any(m.q), g.Q = spdiags(m.q,0,numel(m.q),numel(m.q)); end
if integral
    g.vtype = repmat('C',size(m.obj)); g.vtype(m.binary) = 'B';
end
p = struct('OutputFlag',0,'Threads',1,'TimeLimit',seconds,'MIPGap',1e-6, ...
    'FeasibilityTol',1e-8,'OptimalityTol',1e-8,'BarConvTol',1e-10,'Seed',7);
r = gurobi(g,p);
r.lower = -Inf;
if isfield(r,'objbound') && integral
    r.lower = r.objbound;
elseif strcmp(r.status,'OPTIMAL') && isfield(r,'pi')
    % Pi is RHS sensitivity; lambda >= 0 multiplies A*x-rhs <= 0.
    r.lambda = -r.pi; r.lambda(~m.eq) = max(0,r.lambda(~m.eq));
    c = m.obj + m.A'*r.lambda;
    z = m.lb; z(c < 0) = m.ub(c < 0);
    curved = m.q > 0;
    z(curved) = min(m.ub(curved),max(m.lb(curved),-c(curved)./(2*m.q(curved))));
    if all(isfinite(z))
        r.lower = -m.rhs'*r.lambda + sum(c.*z + m.q.*z.^2)-1e-7;
    end
end
end
