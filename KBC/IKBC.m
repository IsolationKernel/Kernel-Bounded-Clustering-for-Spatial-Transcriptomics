function TTclass = IKBC(ndata,K,tau,k,sID)

%% IKBC：@hang
% ---------------------------------------------------------------
% |  ndata is the feature map of data.                          |
% |  K is the sim matrix of the s points sampled from data.     | 
% |  tau is thyrold of sim.                                     |
% |  k is the number of clusters.                               |  
% |  sID is the index of s points.                              |
% ---------------------------------------------------------------

%% step 1
K_g = K>=tau;
G = graph(K_g);
[Tclass, number] = conncomp(G);    
%%
if size(number,2)>=k
    %% step 2 
    Tclass = Tclass';
    [~,index] = maxk(number,k);
    Tlass_k = zeros([size(ndata,1),1]);
    mean_i = [];
    for i = 1:1:k
        Tlass_k(sID(Tclass==index(i)))=i;
        mean_i = [mean_i;mean(ndata(sID(Tclass==index(i)),:),1)];
    end
    
    %% step 3
    needclass = ndata(Tlass_k==0,:)*mean_i';
    [~,aha]=max(needclass,[],2);
    Tlass_k(Tlass_k==0) = aha;
    TTclass = Tlass_k;
    
%% postprocessing

Tclass = TTclass;
Th=ceil(size(ndata,1)*0.01);
Tclass2=Tclass+1;

for iter=1:100
    Cmean=[];
    for i=1:k
        Cmean=[Cmean;mean(ndata(Tclass==i,:),1)];
    end
    [~,Tclass2]=max(ndata*Cmean',[],2);
    
    if sum(Tclass2~=Tclass)<Th
        break
    end
    Tclass=Tclass2;
end
TTclass = Tclass;

else
    TTclass = zeros(size(ndata,1),1)+1;   
end