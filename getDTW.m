function distanceDTW=getDTW(sample_x,sample_y)
N=length(sample_x);
M=length(sample_y);
DTW=zeros(N+1,M+1);
DTW(2:(N+1),1)=inf;
DTW(1,2:(M+1))=inf;
DTW(1,1)=0;
for i=2:(N+1)
    for j=2:(M+1)
        cost=abs(sample_x(i-1)-sample_y(j-1));
        DTW(i,j)=cost+min([DTW(i-1,j) DTW(i,j-1) DTW(i-1,j-1)]);
    end
end
distanceDTW=DTW(N+1,M+1);
clear DTW;
end