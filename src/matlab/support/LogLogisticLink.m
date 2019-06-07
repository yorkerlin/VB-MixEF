function [varargout] = LogLogisticLink(xSampled,iter,X,y,N,M)
k = size(X,2);
if M==N
    ind = [1:N];
else
	setSeed(iter);
    %ind = unidrnd(N, [M,1]);
    ind = randperm(N,M)';
end

X_hat = full(X(ind,:));
%y_hat = y(ind)>0;
y_hat = (y(ind)-0.5)>0;
y_hat = 2*y_hat-1;%n by 1
assert(size(X_hat,1) == M)

maxBlockSize = ceil(500000000./(M*k));
totalBlockSize = size(xSampled,2);
lp = zeros(totalBlockSize,1);
if nargout>1
    grad = zeros(k,totalBlockSize);
    if nargout>2
        hess = zeros(k,k,totalBlockSize);
    end
end
for i=1:ceil(totalBlockSize/maxBlockSize)
    beginIdx = (i-1)*maxBlockSize + 1;
    endIdx = i*maxBlockSize;
    if endIdx>totalBlockSize
        endIdx = totalBlockSize;
    end

    betaDraws=xSampled(:,beginIdx:endIdx);%k by S
    S = size(betaDraws, 2);

    fn = (X_hat*betaDraws)';%S by n
    hyp.lik = [];
    lik = {@likLogistic};
    % compute MC approximation (code taken from GPML)
    yhat = repmat(y_hat(:)', S, 1);
    if nargout==1
        f = feval(lik{:}, hyp, yhat, fn, [], 'infLaplace');
    else
        [f, df, d2f] = feval(lik{:}, hyp, yhat, fn, [], 'infLaplace');
    end
    assert( size(f',1) == M )
    f = sum(f',1)/M; %lp =  sum_x \log link(x,w)
    lp(beginIdx:endIdx,1) = (f*N)';
    if nargout>1
        gm = df';
        g = X_hat'*gm/M;  %G
        grad(:,beginIdx:endIdx) = g*N;
        if nargout>2
            gv = d2f'/2;
            tmp1 = repmat(X_hat,1,1,S) .* reshape(2*gv,size(gv,1),1,size(gv,2));%diag(2*gv)*X
            H = einsum('ij,jkp->ikp',X_hat',tmp1)/M;
            hess(:,:,beginIdx:endIdx) = H*N;
        end
    end

end

if nargout==1
    varargout = {lp};
else
    if nargout>1
        varargout = {lp, grad};
        if nargout>2
            varargout = {lp, grad, hess};
        end
    end
end

end
