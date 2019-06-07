%function [varargout] = LogLogisticPost(xSampled,iter,X,y,priorMean,priorPrec,priorMeanTimesPrec,alpha,N,M)
%k = size(X,2);
%if M==N
    %ind = [1:N];
%else
    %setSeed(iter);
    %%ind = unidrnd(N, [M,1]);
    %ind = randperm(N,M)';
%end
%alpha = ones(size(priorMean)) .* alpha;
%priorLp = -(  sum( ((xSampled-priorMean).^2)./alpha, 1) + k*log(2*pi) + sum(log(alpha))  )/2;

%X_hat = full(X(ind,:));
%y_hat = y(ind)>0;
%y_hat = 2*y_hat-1;%n by 1
%betaDraws=xSampled;%k by S
%S = size(betaDraws, 2);

%fn = (X_hat*betaDraws)';%S by n
%hyp.lik = [];
%lik = {@likLogistic};
%% compute MC approximation (code taken from GPML)
%y_hat = repmat(y_hat(:)', S, 1);
%if nargout==1
    %f = feval(lik{:}, hyp, y_hat, fn, [], 'infLaplace');
%else
    %[f, df, d2f] = feval(lik{:}, hyp, y_hat, fn, [], 'infLaplace');
%end
%assert( size(f',1) == M )
%f = sum(f',1)/M; %Lp
%lp = (f*N + priorLp)';
%varargout = {lp};
%if nargout>1
    %gm = df';
    %g = X_hat'*gm/M;  %G
    %grad = g*N + priorMeanTimesPrec - priorPrec*xSampled;
    %varargout = {lp, grad};
    %if nargout>2
        %gv = d2f'/2;
        %tmp1 = repmat(X_hat,1,1,S) .* reshape(2*gv,size(gv,1),1,size(gv,2));%diag(2*gv)*X
        %H = einsum('ij,jkp->ikp',X_hat',tmp1)/M;
        %hess = H*N - priorPrec;
        %varargout = {lp, grad, hess};

        %[l2, g2, h2] = LogLogisticPost2(xSampled,iter,X,y,priorMean,priorPrec,priorMeanTimesPrec,alpha,N,M);

        %sum(abs( lp(:) - l2(:) ))
        %sum(abs( grad(:) - g2(:) ))
        %sum(abs( hess(:) - h2(:) ))

    %end
%end
%end

function [varargout] = LogLogisticPost(xSampled,iter,X,y,priorMean,priorPrec,priorMeanTimesPrec,alpha,N,M)
k = size(X,2);
if M==N
    ind = [1:N];
else
	setSeed(iter);
    %ind = unidrnd(N, [M,1]);
    ind = randperm(N,M)';
end
alpha = ones(size(priorMean)) .* alpha;
priorLp = -(  sum( ((xSampled-priorMean).^2)./alpha, 1) + k*log(2*pi) + sum(log(alpha))  )/2;

X_hat = full(X(ind,:));
%y_hat = y(ind)>0;
y_hat = (y(ind)-0.5)>0;
y_hat = 2*y_hat-1;%n by 1

maxBlockSize = ceil(500000000./(M*k));
totalBlockSize = size(xSampled,2);
if nargout==1
    lp = zeros(totalBlockSize,1);
else
    if nargout>1
        grad = zeros(k,totalBlockSize);
        if nargout>2
            hess = zeros(k,k,totalBlockSize);
        end
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
    f = sum(f',1)/M; %Lp
    lp(beginIdx:endIdx,1) = (f*N + priorLp(1,beginIdx:endIdx))';
    if nargout>1
        gm = df';
        g = X_hat'*gm/M;  %G
        grad(:,beginIdx:endIdx) = g*N + priorMeanTimesPrec - priorPrec*betaDraws;
        if nargout>2
            gv = d2f'/2;
            tmp1 = repmat(X_hat,1,1,S) .* reshape(2*gv,size(gv,1),1,size(gv,2));%diag(2*gv)*X
            H = einsum('ij,jkp->ikp',X_hat',tmp1)/M;
            hess(:,:,beginIdx:endIdx) = H*N - priorPrec;
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
