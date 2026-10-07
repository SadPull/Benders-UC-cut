function x = repair(d,s,u)
%REPAIR Deterministic temporal/reserve repair; dispatch must still be checked.
x=[];
if ~isequal(size(u),[d.n_units d.horizon]) || any(~isfinite(u),'all'), return; end
u=min(1,max(0,round(u)));
for i=1:d.n_units
    state=d.initial_on(i); duration=d.initial_duration(i);
    for t=1:d.horizon
        minimum=state*d.min_up(i)+(1-state)*d.min_down(i);
        target=u(i,t);
        if target~=state && (duration<minimum || (t==d.horizon && state==1)), target=state; end
        if target==state, duration=duration+1; else, duration=1; end
        state=target; u(i,t)=state;
    end
end
required=(1+d.reserve_fraction)*max(vertcat(s.load),[],1);
for iteration=1:d.n_units+1
    [candidate,deficit]=pack(u);
    if all(deficit<=1e-6), x=candidate; return; end
    t=find(deficit>1e-6,1); winner=[]; best_gain=0; best_cost=Inf;
    for i=1:d.n_units
        if d.initial_on(i), start=1;
        else, start=max(1,d.min_down(i)-d.initial_duration(i)+1); end
        if start>d.horizon, continue; end
        trial=u; trial(i,start:end)=1;
        [v,remaining]=pack(trial); gain=deficit(t)-remaining(t);
        % x is grouped by variable type, then MATLAB column-major unit/time.
        cost=[repmat(d.cost_c,d.horizon,1);repmat(d.startup_cost,d.horizon,1); ...
            repmat(d.shutdown_cost,d.horizon,1)]'*v;
        if all(remaining<=deficit+1e-6) && gain>1e-6 && ...
                (gain>best_gain+1e-9 || (abs(gain-best_gain)<=1e-9 && cost<best_cost))
            winner=trial; best_gain=gain; best_cost=cost;
        end
    end
    if isempty(winner), return; end
    u=winner;
end
    function [z,deficit]=pack(on)
        previous=[d.initial_on on(:,1:end-1)];
        v=max(on-previous,0); w=max(previous-on,0);
        capacity=min(d.pmax.*on-(d.pmax-d.pstart).*v, ...
            d.pmax.*on-(d.pmax-d.pshut).*[w(:,2:end) zeros(d.n_units,1)]);
        deficit=max(0,required-sum(capacity,1)); z=[on(:);v(:);w(:)];
    end
end
