function [V_final,K,Qf] = CSNMF(W,mu,k,Z,alpha)


maxIter=500;

n=size(W,2);
Os=ones(n,1);
nIter=0;
num=1;
tryNo=0;
obj=1e+7;
 
% ------KNN graph 
graphD=zeros(size(W,1),size(W,1));

GD = full(sum(W,2));
graphD = spdiags(GD,0,n,n);
graphL=graphD-W;
% ------Norm GraphL
D_mhalf = spdiags(GD.^-.5,0,n,n) ;
graphL = D_mhalf*graphL*D_mhalf;


while tryNo < 5 
    tryNo = tryNo+1;

% ------initialize U V K
Vb=rand(n,k);
Kb=rand(n);


[Vb] = NormalizeK(Vb);
[Kb] = NormalizeK(Kb);

for i=1:n
    ind=find(Z(i,:)==1);
    Kb(i,ind)=1;
end

% ------kernel graph
kerD=zeros(size(Kb,1),size(Kb,1));
W2=(Kb+Kb')/2;

FRb=norm(W2-Vb*Vb')+mu*trace(graphL*Kb)+alpha*norm(Kb-Z);


while nIter<maxIter
    
    % -----update K
    
    VV=Vb*Vb';
    
    T2=(2*alpha+1)*Kb+Kb'+mu*graphD;
    T1=2*VV+2*alpha*Z'+mu*W;
          
    K=Kb.*(T1./T2);    
    Kb=K;
        
    W2=(Kb+Kb')/2;
   
    % -----update V

    C=W2*Vb;
    D=Vb*Vb'*Vb;
    E=C./D;
    V=Vb.*(E.^(1/4));
    Vb=V;
      
%     FR=norm(W2-Vb*Vb')+mu*trace(graphL*Kb)+alpha*norm(Kb-Z);
%     Qf(num,1)=FR;
%     num=num+1;
    nIter=nIter+1;
end

FR=norm(W2-Vb*Vb')+mu*trace(graphL*Kb)+alpha*norm(Kb-Z);

if FR < obj
    [V_final] = NormalizeK(V);
    obj=FR;
    nIter=0;

end

end

   
    function [K] = NormalizeK(K)
        n = size(K,2);
            norms = max(1e-15,sqrt(sum(K.^2,1)))';
            K = K*spdiags(norms.^-1,0,n,n);


