clear
clc

resulttop =  strings(1,1);
parameterset = zeros(1,1);


mat_data = load('../Dataset/Stereo-seq/wl/sterseq_normwl.mat');
data = mat_data.data;
pos = mat_data.pos;
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
psilist=[8];
psilist=psilist(psilist<size(data,1));
Klist=[0.5];
t=100; % number of partitioning
rounds = 10;
s=min(size(data,1),1000);   %sample size 
%--------------------------------------------------------------------------

%% search the best NMI and F1
AA = [];
FA = [];
RA = [];
LABEList = [];
for i =1:1:rounds
for pp=1:length(psilist)
    psi=psilist(pp); 
    ndata = iNNEspace_zjdis_fast(data,data,psi,t,distance_all);
    sID = randperm(size(ndata,1),s);  
    K = ndata(sID,:)*ndata(sID,:)'./t;
    AA_psi = [];
    FA_psi = [];
    ARI_psi = [];
    refARI_psi = [];
    labellist = [];
    reflabellist = [];
    for tt=1:length(Klist)  
        Kn=Klist(tt); 
        Tclass = IKBC(ndata,K,Kn,8,sID);

        %Tclass = refineMethod(pos,Tclass,"hexagon");
        save(['../Dataset/Stereo-seq/',num2str(psi),'_',num2str(Kn),'_','.mat'],"Tclass");
        labellist = [labellist,Tclass];
    end
end
end
end
