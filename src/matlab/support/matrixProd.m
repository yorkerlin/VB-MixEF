function res = matrixProd(logw,x)
assert( size(logw,2) == size(x,1) )
res = zeros( size(logw,1) , size(x,2) );

for i=1:size(logw,1)
    for j=1:size(x,2)
        sigInd = sign(x(:,j));
        tmp=sigInd.*exp(logw(i,:)'+log(abs(x(:,j))));
        tmp(sigInd==0) = 0;
        res(i,j)=sum(tmp);
    end
end

end
