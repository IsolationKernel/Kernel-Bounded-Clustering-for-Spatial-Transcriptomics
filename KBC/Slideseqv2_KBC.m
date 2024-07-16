clear
clc


mat_data = load('../Dataset/SlideseqV2hippocampus/wl/slideseqv2h5pc20.mat');
data = mat_data.data;
pos = mat_data.pos;

%% data normalisation
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
distance_all = pdist2(data,data,'cosine');
for runtimes = 1:1

%% parameter setting
%--------------------------------------------------------------------------
psi=6;
Kn=0.5;
t=100; % number of partitioning
k=14;
rounds = 1;
s=min(size(data,1),1000);   %sample size 
%--------------------------------------------------------------------------


LABEList = [];
for i =1:1:rounds 
    ndata = iNNEspace_zjdis_fast(data,data,psi,t,distance_all);
%--------------------------------------------------------------------------
    sID = randperm(size(ndata,1),s);
%+++++++++++++++++++++++++++sim matrix based on euc distance+++++++++++++++
%     K = pdist2(ndata(sID,:),ndata(sID,:));
%     K = 1 - K./max(max(K));
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%---------------------------sim matrix based on dot product----------------  
    K = ndata(sID,:)*ndata(sID,:)'./t;
    refARI_psi = [];
    labellist = [];
    reflabellist = [];
    Tclass = IKBC(ndata,K,Kn,k,sID);
    save('../Dataset/SlideseqV2hippocampus/labels.mat',"Tclass");

end

end

