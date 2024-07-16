clear
clc
maindir = 'DataSet/MousePosterior';
subdir  = dir( maindir );
resultmean = strings(12,1);
resulttop =  strings(12,1);
parameterset = zeros(12,1);

for filenum = 1: 12
load( [maindir,'/',subdir(filenum+2).name])
class = double(class);
%% the min index of class is 1
% -------------------------------------------------------------------------
min(class);
if min(class)==-1
    class = class>0;
end
min(class)
if min(class)==0
    class = class + 1;
    disp("please check min k");
end

%--------------------------------------------------------------------------
%% data normalisation
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
distance_all = pdist2(data,data,'cosine');
for runtimes = 1:1

%% parameter setting
%--------------------------------------------------------------------------
psilist=[2 4 6 8 16 24 32 48 64 80 100 200 250 512];
psilist=psilist(psilist<size(data,1));
Klist=[0.05:0.05:0.95];
t=100; % number of partitioning
k=size(unique(class),1);
rounds = 10;
s=min(size(data,1),1000);   %sample size 
%--------------------------------------------------------------------------

%% search the best NMI and F1
AA = [];
FA = [];
RA = [];
LABEList = [];
for i =1:1:rounds
parfor pp=1:length(psilist)
    psi=psilist(pp); 
    ndata = iNNEspace_zjdis_fast(data,data,psi,t,distance_all);
%--------------------------------------------------------------------------
    sID = randperm(size(ndata,1),s);
%+++++++++++++++++++++++++++sim matrix based on euc distance+++++++++++++++
%     K = pdist2(ndata(sID,:),ndata(sID,:));
%     K = 1 - K./max(max(K));
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%---------------------------sim matrix based on dot product----------------  
    K = ndata(sID,:)*ndata(sID,:)'./t;
%--------------------------------------------------------------------------
%% there are two FA and AA
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% | first, if you need the best parameter psi and tau, you can use the max| 
% | average of round 10                                                   |
% | second, if you just need the best NMI and F1, just search the best    |
% | value for different psi, because the feature map is different for  d- |
% | ifferent psi, compute the average of different tau for rounds 10 is   |
% | incorrect.                                                            |
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    AA_psi = [];
    FA_psi = [];
    ARI_psi = [];
    refARI_psi = [];
    labellist = [];
    reflabellist = [];
    for tt=1:length(Klist)  
        Kn=Klist(tt); 
        Tclass = IKBC(ndata,K,Kn,k,sID);
        [NMI] = ami(class,Tclass);
        [F1,recall,acc]=fmeasure(class,Tclass);
        [ARI,RI,MI,HI]=RandIndex(class,Tclass);
%         AA(pp,tt)=NMI;
%         FA(pp,tt)=F1;
        AA_psi = [AA_psi,NMI];
        FA_psi = [FA_psi,acc];
        ARI_psi = [ARI_psi,ARI];
        labellist = [labellist,Tclass];
    end
    AA(i,pp)=max(AA_psi);
    FA(i,pp)=max(FA_psi);
    [ARI_imax,ari_max_index]=max(ARI_psi);
    RA(i,pp)=ARI_imax;
    LABEList = [LABEList,labellist(:,ari_max_index)];
end
end

% AA = BB./rounds;
% FA = FB./rounds;

bestNMI=max(mean(AA));% the best/ performance
bestACC=max(mean(FA));
bestARI=max(mean(RA));
[arimax1,index1] = max(RA);
[arimax2,index2] = max(max(RA));
index_label = (index1(index2)-1)*size(psilist,2) + index2;
bestLabel = LABEList(:,index_label);
[ARI,RI,MI,HI]=RandIndex(class,bestLabel);
resultmean(filenum,runtimes) = num2str(bestARI);
parameterset(filenum,1) = index2
bestLabel = refineMethod(pos,bestLabel,"hexagon");
[ARI,RI,MI,HI]=RandIndex(class,bestLabel);
resulttop(filenum,runtimes)  = num2str(ARI);

end
end
