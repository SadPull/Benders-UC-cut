function report = validate(d,s,result)
%VALIDATE Independent physical equations and objective reconstruction.
assert(isfinite(result.objective),'suc:Validation','No finite objective.');
for name={'u','v','w'}
    z=result.(name{1});
    assert(isequal(size(z),[d.n_units d.horizon]) && all(z==0 | z==1,'all'), ...
        'suc:Validation','Invalid binary schedule.');
end
u=result.u; v=result.v; w=result.w;
previous=[d.initial_on u(:,1:end-1)];
assert(isequal(v,max(u-previous,0)) && isequal(w,max(previous-u,0)), ...
    'suc:Validation','Startup/shutdown transition mismatch.');
assert(all(w(:,end)==0),'suc:Validation','Terminal shutdown is forbidden.');
for i=1:d.n_units
    state=d.initial_on(i); duration=d.initial_duration(i);
    for t=1:d.horizon
        minimum=state*d.min_up(i)+(1-state)*d.min_down(i);
        if u(i,t)~=state
            assert(duration>=minimum,'suc:Validation','Minimum up/down time violated.');
            state=u(i,t); duration=1;
        else
            duration=duration+1;
        end
    end
end
assert(numel(result.scenarios)==numel(s),'suc:Validation','Missing scenarios.');
startup=d.pmax.*u-(d.pmax-d.pstart).*v;
shutdown=d.pmax.*u-(d.pmax-d.pshut).*[w(:,2:end) zeros(d.n_units,1)];
maximum=0; fuel=0;
for k=1:numel(s)
    z=result.scenarios(k); p=z.p; r=z.r;
    assert(strcmp(z.name,s(k).name) && isequal(size(p),size(u)) && isequal(size(r),size(u)) ...
        && all(isfinite(p),'all') && all(isfinite(r),'all'),'suc:Validation','Invalid dispatch.');
    previous=[d.initial_output p(:,1:end-1)];
    residual=[abs(sum(p,1)-s(k).load),d.reserve_fraction*s(k).load-sum(r,1), ...
        reshape(d.pmin.*u-p,1,[]),reshape(p+r-startup,1,[]),reshape(p+r-shutdown,1,[]), ...
        reshape(p+r-d.pmax.*u,1,[]),reshape(p-previous-d.ramp_up,1,[]), ...
        reshape(previous-p-d.ramp_down,1,[]),-p(:)',-r(:)'];
    maximum=max([maximum residual]);
    cost=sum(d.cost_a.*p.^2+d.cost_b.*p,'all');
    assert(abs(cost-z.objective)<=1e-4,'suc:Validation','Scenario fuel cost mismatch.');
    fuel=fuel+s(k).probability*cost;
end
fixed=sum(d.cost_c.*u+d.startup_cost.*v+d.shutdown_cost.*w,'all');
error_cost=abs(fixed+fuel-result.objective);
assert(maximum<=1e-5 && error_cost<=1e-4,'suc:Validation','Physical or total cost validation failed.');
report=struct('feasible',true,'max_residual_mw',maximum,'cost_residual',error_cost);
end
