function m = model(d,s)
%MODEL Whole-horizon sparse SUC. Every row is A*x <= rhs or == rhs.
n = d.n_units; T = d.horizon; S = numel(s); N = n*T;
ix.u = reshape(1:N,n,T); ix.v = ix.u+N; ix.w = ix.u+2*N;
ix.p = reshape(3*N+(1:N*S),n,T,S); ix.r = ix.p+N*S;
nv = (3+2*S)*N;
m.obj = zeros(nv,1); m.q = zeros(nv,1); m.lb = zeros(nv,1); m.ub = ones(nv,1);
m.obj(ix.u) = repmat(d.cost_c,1,T); m.obj(ix.v) = repmat(d.startup_cost,1,T);
m.obj(ix.w) = repmat(d.shutdown_cost,1,T);
for k = 1:S
    m.obj(ix.p(:,:,k)) = repmat(s(k).probability*d.cost_b,1,T);
    m.q(ix.p(:,:,k)) = repmat(s(k).probability*d.cost_a,1,T);
    m.ub(ix.p(:,:,k)) = repmat(d.pmax,1,T); m.ub(ix.r(:,:,k)) = repmat(d.pmax,1,T);
end
m.binary = (1:3*N)';
m.time = repmat(repelem((1:T)',n),3+2*S,1);
I = {}; J = {}; V = {}; rhs = []; eq = []; labels = {};
for t = 1:T
    for i = 1:n
        u = ix.u(i,t); v = ix.v(i,t); w = ix.w(i,t);
        if t == 1
            add([u v w],[1 -1 1],d.initial_on(i),true,'transition');
        else
            add([u v w ix.u(i,t-1)],[1 -1 1 -1],0,true,'transition');
        end
        add([v w],[1 1],1,false,'exclusive');
        if t == T, add(w,1,0,true,'terminal'); end
        remaining = d.initial_on(i)*d.min_up(i)+(1-d.initial_on(i))*d.min_down(i)-d.initial_duration(i);
        if t <= remaining, add(u,1,d.initial_on(i),true,'initial'); end
        for j = t:min(T,t+d.min_up(i)-1)
            add([v ix.u(i,j)],[1 -1],0,false,'min_up');
        end
        for j = t:min(T,t+d.min_down(i)-1)
            add([w ix.u(i,j)],[1 1],1,false,'min_down');
        end
        for k = 1:S
            p = ix.p(i,t,k); r = ix.r(i,t,k);
            add([u p],[d.pmin(i) -1],0,false,'pmin');
            add([p r u v],[1 1 -d.pmax(i) d.pmax(i)-d.pstart(i)],0,false,'startup_capacity');
            add([p r u],[1 1 -d.pmax(i)],0,false,'envelope');
            if t < T
                add([p r u ix.w(i,t+1)],[1 1 -d.pmax(i) d.pmax(i)-d.pshut(i)],0,false,'shutdown_capacity');
            else
                add([p r u],[1 1 -d.pmax(i)],0,false,'shutdown_capacity');
            end
            if t == 1
                add(p,1,d.ramp_up(i)+d.initial_output(i),false,'ramp_up');
                add(p,-1,d.ramp_down(i)-d.initial_output(i),false,'ramp_down');
            else
                add([p ix.p(i,t-1,k)],[1 -1],d.ramp_up(i),false,'ramp_up');
                add([p ix.p(i,t-1,k)],[-1 1],d.ramp_down(i),false,'ramp_down');
            end
        end
    end
    for k = 1:S
        add(ix.p(:,t,k),ones(n,1),s(k).load(t),true,'balance');
        add(ix.r(:,t,k),-ones(n,1),-d.reserve_fraction*s(k).load(t),false,'reserve');
    end
end
m.A = sparse(vertcat(I{:}),vertcat(J{:}),vertcat(V{:}),numel(rhs),nv);
m.rhs = rhs(:); m.eq = logical(eq(:)); m.labels = labels(:); m.ix = ix;
m.d = d; m.scenarios = s; m.constant = 0;
    function add(j,v,b,e,label)
        keep = v ~= 0; j = j(keep); v = v(keep); row = numel(rhs)+1;
        I{row} = repmat(row,numel(j),1); J{row} = j(:); V{row} = v(:);
        rhs(row) = b; eq(row) = e; labels{row} = label;
    end
end
