function [ndata] = iNNEspace_zjdis(Sdata,data, psi, t)

[sn,~]=size(Sdata);
[n,~]=size(data);

ndata=[];
c=0:psi+1:(n-1)*(psi+1);

for i = 1:t
    % sampling    
    CurtIndex = datasample(1:sn, psi, 'Replace', false);
    Ndata = Sdata(CurtIndex,:);
    
    % filter out repeat
    
    [~,IA,~] = unique(Ndata,'rows'); %  C = A(IA) and A = C(IC) (or A(:) = C(IC), if A is a matrix or array).
    NCurtIndex=CurtIndex(IA);
    Ndata = data(NCurtIndex,:);
    
    % radius
%     [D,~] = pdist2(Ndata,Ndata,'minkowski',2,'Smallest',2);    
%     R=D(2,:);  % radius
%     
%     % identify 1NN ball for each point    
%     [D,I] = pdist2(Ndata,data,'minkowski',2,'Smallest',1);
%     I(D>R(I))=psi+1; % outside ball

%     sample_sample =Ndata.^2*ones(size(Ndata'))+ones(size(Ndata))*(Ndata').^2-2*Ndata*Ndata';
    sample_sample = pdist2_fast(Ndata,Ndata,'euclidean');
    R=mink(sample_sample,2,2)';
    R = R(2,:);
%     distance_sample = Ndata.^2*ones(size(data'))+ones(size(Ndata))*(data').^2-2*Ndata*data';
    distance_sample = pdist2_fast(Ndata,data,'euclidean');
    [D,I]=min(distance_sample,[],1);
    I(D>R(I))=psi+1; % outside ball
    
    z=zeros(psi+1,n);
    z(I+c)=1;
    z(psi+1,:)=[]; % get rid of values that outside ball
    ndata=[ndata sparse(z)'];    
end
end