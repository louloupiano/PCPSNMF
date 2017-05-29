clc;
clear;
load PIE

fea = NormalizeFea(fea');
fulldata = fea';%fulldata=D*N
Trans_fulldata=fea;

for k=2:1:10
    num=1;
    disp(sprintf('# of classes: %g',k))
    for nter=1:10
        %% generate data and label's subset 
        per = 5;
        [datasub,subN,labelsub,Trans_datasub,clist]=loadsub(fulldata,gnd,k);
        labeled_num=per*k;
        
        %% W
        nnparams=cell(1);
        nnparams{1}='knn';
        opts.K =4; 
        opts.maxblk = 1e7;
        opts.metric = 'eucdist';
        nnparams{2}=opts;
        T_G=slnngraph(datasub,[],nnparams);
        
        W=zeros(subN);
        for i = 1:subN
            ind = find(T_G(:,i)~=0);
            W(ind,i) = exp(-0.25*T_G(ind,i));  
        end
        W=(W+W')/2;
        
        % construct affinity matrix (D^-1/2)*W*(D^-1/2)
        GD = full(sum(W,2));
        D_mhalf = spdiags(GD.^-.5,0,subN,subN);
        graphL = D_mhalf*W*D_mhalf;
        
        %% each datasub repeats 10 trails
        
        % separate into train and test datasets
        indices=findlabels(per,labelsub,k);
        traindata = Trans_datasub(indices,:);
        trainlabel = labelsub(indices);
        ind=[1:1:subN];
        ind(indices)=[];
        testdata= Trans_datasub(ind,:);
        testlabel=labelsub(ind);
        testnum=size(testdata,1);
        
        % prior constraint
        Z=eye(subN);
        for i=1:k
            temp=indices((i-1)*per+1:i*per,1);
            for j=1:per
                Z(temp(j),temp)=1;
            end
        end

        %% PCPSNMF
        mu=0.5;
        alpha=1;%optimal
        
        [Vn,K] = CSNMF(W,mu,k,Z,alpha);
        [tmp label] = max(Vn, [], 2); 
        label(indices)=[];
        label = bestMap(testlabel,label);
            
        AC_our(num,k-1) = length(find(testlabel == label))/length(testlabel);
        MIhat_our(num,k-1) = MutualInfo(testlabel,label);
        ac1=AC_our(num,k-1);
        nmi1=MIhat_our(num,k-1);       
                
        %% GSNMF        
        options = [];
        options.maxIter = 500;
        options.alpha = 1;
        options.nRepeat = 5;
        options.triF = 0; %bi factorization
        
        [S,H] = GNMF_S(graphL,k,Z,options);
        [tmp label2] = max(H, [], 2); 
        label2(indices)=[];
        label2 = bestMap(testlabel,label2);
        AC_snmf(num,k-1) = length(find(testlabel == label2))/length(testlabel);
        MIhat_snmf(num,k-1) = MutualInfo(testlabel,label2);
        
        %% separate SNMF PCP
        alpha = 1;
        [Ker] = KerP(W,alpha,Z);
        
        options = [];
        options.maxIter = 500;
        options.alpha = 0;
        options.nRepeat = 5;
        [S,Hstep] = GNMF_S(Ker,k,Z,options);
        
        [tmp label3] = max(Hstep, [], 2); 
        label3(indices)=[];
        label3 = bestMap(testlabel,label3);
        AC_sepSNMF(num,k-1) = length(find(testlabel == label3))/length(testlabel);
        MIhat_sepSNMF(num,k-1) = MutualInfo(testlabel,label3);         
        
        %% GNMF                
        options = [];
        options.maxIter = 500;
        options.alpha = 100;
        options.nRepeat = 5;
        [S,gH] = GNMF(datasub,k,W,options);

        label4 = litekmeans(gH,k,'Replicates',20);
        label4(indices)=[];
        label4 = bestMap(testlabel,label4);        

        AC_GNMF(num,k-1) = length(find(testlabel == label4))/length(testlabel);%ac4;
        MIhat_GNMF(num,k-1) = MutualInfo(testlabel,label4);%nmi4;
        
        
        %% Kmeans        
        label5 = litekmeans(testdata,k,'Replicates',20);
        label5 = bestMap(testlabel,label5);
        AC_Kmeans(num,k-1) = length(find(testlabel == label5))/length(testlabel);
        MIhat_Kmeans(num,k-1) = MutualInfo(testlabel,label5);
        
        %% CNMF
        [S,Hc] = ConstainedNMF(traindata,testdata,indices,trainlabel,k);
        label6 = litekmeans(Hc,k,'Replicates',20);
        label6(1:k*per)=[];
        label6 = bestMap(testlabel,label6);
                
        AC_Cnmf(num,k-1) = length(find(testlabel == label6))/length(testlabel);
        MIhat_Cnmf(num,k-1) = MutualInfo(testlabel,label6);
        
         %% SNMFCC        
        A=zeros(subN);        
        for i=1:size(indices)
            for j=1:size(indices)
                A(indices(i),indices(j))=1;                
            end
        end        
        for i=1:k
            temp=indices((i-1)*per+1:i*per,1);
            for j=1:per
                A(temp(j),temp)=6;
            end
        end
        A=A-diag(diag(A));
        
        Hscc = SNMFCC(graphL,A,Z,k);
        [tmp label7] = max(Hscc, [], 2); 
        label7(indices)=[];
        label7 = bestMap(testlabel,label7);
        AC_SNMFCC(num,k-1) = length(find(testlabel == label7))/length(testlabel);
        MIhat_SNMFCC(num,k-1) = MutualInfo(testlabel,label7); 
        
        num=num+1;
        disp(sprintf('Iteration %g finished',nter))    
    end
end

%% plot figure
figure
x=[2:1:10]*1; 
subplot(1,2,1)

ans = mean(AC_our,1);
bns = mean(AC_snmf,1);
cns = mean(AC_sepSNMF,1);
dns = mean(AC_GNMF,1);
ens = mean(AC_Cnmf,1);
fns = mean(AC_Kmeans,1);
gns = mean(AC_SNMFCC,1);

plot(x,ans,'-b^','LineWidth', 2);
hold on
plot(x,bns,'-m*','LineWidth', 2);
hold on
plot(x,cns,'-rd','LineWidth', 2);
hold on
plot(x,dns,'-gv','LineWidth', 2);
hold on
plot(x,ens,'-ks','LineWidth', 2);
hold on
plot(x,fns,'-co','LineWidth', 2);
hold on
plot(x,gns,'-yo','LineWidth', 2);
hold off
grid
title('PIE Dataset')
xlabel('Number of Classes')
ylabel('Accuracy')

subplot(1,2,2)

ans = mean(MIhat_our,1);
bns = mean(MIhat_snmf,1);
cns = mean(MIhat_sepSNMF,1);
dns = mean(MIhat_GNMF,1);
ens = mean(MIhat_Cnmf,1);
fns = mean(MIhat_Kmeans,1);
gns = mean(MIhat_SNMFCC,1);

plot(x,ans,'-b^','LineWidth', 2);
hold on
plot(x,bns,'-m*','LineWidth', 2);
hold on
plot(x,cns,'-rd','LineWidth', 2);
hold on
plot(x,dns,'-gv','LineWidth', 2);
hold on
plot(x,ens,'-ks','LineWidth', 2);
hold on
plot(x,fns,'-co','LineWidth', 2);
hold on
plot(x,gns,'-yo','LineWidth', 2);
hold off
grid
title('PIE Dataset')
xlabel('Number of Classes')
ylabel('NMI')
h = legend('Proposed method','GSNMF','PCP+SNMF','GNMF','CNMF','Kmeans','SNMFCC','Location','northwest');


