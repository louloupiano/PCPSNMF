function [U,V_final,Qf] = ConstainedNMF(traindata,testdata,indices,trainlabel,k)


X=[traindata;testdata]';
X=X+0.00000001*ones(size(X,1),size(X,2));
m=size(X,1);
n=size(X,2);
nl=size(indices,1);
A = zeros(n,k+n-nl);
A(nl+1:end,k+1:end)=eye(n-nl);
for i=1:nl
    A(i,trainlabel(i))=1;
end
    
maxIter=500;

Os=ones(n,1);
nIter=0;
num=1;
tryNo=0;
obj=1e+5;

while tryNo <5  
    tryNo = tryNo+1;
% ------initialize U Z
Ub=rand(m,k);
[Ub] = NormalizeK(Ub);
Zb=rand(size(A,2),k);
[Zb] = NormalizeK(Zb);

FRb=norm(X-Ub*Zb'*A');
% eval=1;

while nIter<maxIter
    
    % -----update U
    C=X*A*Zb;
    D=Ub*Zb'*A'*A*Zb;
    U=Ub.*(C./D);
    Ub=U;

    % -----update Z
    E=A'*X'*Ub;
    F=A'*A*Zb*Ub'*Ub;
    Z=Zb.*(E./F);    
    Zb=Z;

    
%     FR=norm(X-Ub*Zb'*A');
    nIter=nIter+1;
%     Qf(nIter,1)=FR;
    
end
V=A*Z;
FR=norm(X-Ub*Zb'*A');
if FR < obj
    V_final = V;
    obj=FR;
    nIter=0;
end

end



function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end
    
    function [K] = NormalizeK(K)
        n = size(K,2);
            norms = max(1e-15,sqrt(sum(K.^2,1)))';
            K = K*spdiags(norms.^-1,0,n,n);


