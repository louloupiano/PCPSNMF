function [datasub,subN,labelsub,Trans_datasub,clist]=loadsub(fulldata,gnd,k)



Dim2 = size(fulldata,1);
datanum = size(fulldata,2);
Class = max(gnd);
turelabel = gnd;
t=find(turelabel==1);

p=randperm(Class);
clist=p(1:k);
%     clist=[19,4,3,17,6,10,18,9,11,16];
% clist=[8,3,6,16,12,20,15,9,7,17];
% clist=[28,2,4,27,40,38,33,18,10,34];

sublist=[];
for i=1:k
    temp=find(turelabel==clist(i));
    sublist=[sublist;temp];
end

temp=ones(size(t,1),1);
labelsub=[];
for i=1:k
    labeltemp=i*temp;
    labelsub=[labelsub;labeltemp];
end

datasub=fulldata(:,sublist);
Trans_datasub=datasub';
    
subN=k*size(t,1);


