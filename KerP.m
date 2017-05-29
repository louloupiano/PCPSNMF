function [K] = KerP(W,alpha,Z)

n=size(W,2);
% ------KNN graph 
graphD=zeros(size(W,1),size(W,1));

GD = full(sum(W,2));
% GD=GD+0.000000001*ones(size(GD,1),1);
graphD = spdiags(GD,0,n,n);
graphL=graphD-W;
%norm GraphL
D_mhalf = spdiags(GD.^-.5,0,n,n) ;
graphL = D_mhalf*graphL*D_mhalf;

K=(2*alpha*Z-graphL)/(2*alpha);

% construct affinity matrix (D^-1/2)*K*(D^-1/2)
    GD = full(sum(K,2));
    D_mhalf = spdiags(GD.^-.5,0,n,n) ;
    graphL = D_mhalf*K*D_mhalf;
    K=graphL;