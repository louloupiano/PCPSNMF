function [H_final,Qf] = SNMFCC(S,A,Q,k)

maxIter=500;

n=size(S,2);
Os=ones(n,1);
nIter=0;
num=1;
tryNo=0;
obj=1e+5;

while tryNo < 5  
    tryNo = tryNo+1;

% ------initialize U V K
Hb=rand(n,k);
[Hb] = NormalizeK(Hb);

FRb=norm(S-Hb*Hb')+norm(A.*(Hb*Hb'-Q));


while nIter<maxIter
    
    % -----update H
    C=S*Hb+(A.*Q.*A')*Hb;
    D=Hb*Hb'*Hb+(A.*(Hb*Hb').*A')*Hb;
    E=(C./D);
    H=Hb.*(E.^(1/4));
    Hb=H;
    Hb(:,1)=Hb(:,1)+1e-30*Os;
    
%     FR=norm(S-Hb*Hb')+norm(A.*(Hb*Hb'-Q));
%     Qf(num,1)=FR;
    num=num+1;

    nIter=nIter+1;
    
end

FR=norm(S-Hb*Hb')+norm(A.*(Hb*Hb'-Q));

if FR < obj
    H_final = Hb;
    obj=FR;
    nIter=0;
end

end

% [V_final] = NormalizeK(V);

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
