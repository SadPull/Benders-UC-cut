function [lambda,upper] = coordinate(planes,equality,incumbent,seconds)
%COORDINATE Maximize the upper cutting-plane model of the concave dual.
nc=numel(equality); B=numel(planes); A=sparse(0,nc+B); rhs=[];
for j=1:B
    assert(~isempty(planes{j}),'suc:Coordination','Each block needs a feasible plane.');
    for k=1:numel(planes{j})
        p=planes{j}(k); row=sparse(1,nc+B);
        row(1:nc)=-p.slope; row(nc+j)=1;
        A(end+1,:)=row; rhs(end+1,1)=p.intercept; %#ok<AGROW>
    end
end
A(end+1,:)=[sparse(1,nc) ones(1,B)]; rhs(end+1,1)=incumbent;
prices_lb=zeros(nc,1); prices_lb(equality)=-Inf; lb=[prices_lb;-Inf(B,1)];
m=struct('A',A,'rhs',rhs,'eq',false(size(rhs)),'obj',[zeros(nc,1);-ones(B,1)], ...
    'q',zeros(nc+B,1),'lb',lb,'ub',Inf(nc+B,1),'binary',[]);
r=suc.optimize(m,seconds);
assert(strcmp(r.status,'OPTIMAL'),'suc:Coordination','Coordination LP failed: %s',r.status);
lambda=r.x(1:nc); lambda(~equality)=max(0,lambda(~equality)); upper=-r.objval;
end
