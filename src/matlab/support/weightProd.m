function wx = weightProd(logW,x)
    if length(logW)>1
        assert( all( size(logW) == size (x) ) )
    end
    tmp = x;
    sigInd = sign(tmp);
    wx=sigInd.*exp(logW+log(abs(tmp)));
    wx(sigInd==0)=0;
end

