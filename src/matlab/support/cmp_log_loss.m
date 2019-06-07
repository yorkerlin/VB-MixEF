function loss = cmp_log_loss(noise,y_te,X_te)
    assert( size(noise,1) == size(X_te,2) )
    assert( size(y_te,1) == size(X_te,1) )

    %y_te is either 0 or 1
    y_te = (y_te-0.5)>0;

    %z = X_te*noise;
    %p_hat2 = mean(sigmoid(z), 2);

    N = size(X_te,1);
    maxBatchSize = 50000;
    p_hat = zeros(N,1);
    for i=1:ceil(N/maxBatchSize)
        beginIdx = (i-1)*maxBatchSize + 1;
        endIdx = i*maxBatchSize;
        if endIdx>N
            endIdx =N;
        end
        z = X_te(beginIdx:endIdx,:)*noise;
        p_hat(beginIdx:endIdx,1) = mean(sigmoid(z), 2);
    end
    loss.log_loss = compute_log_loss_helper(y_te, p_hat);


    %log_lik_1 = -log1p( exp(-z) );
    %log_lik_2 = -z+log_lik_1;
    %tmp = bsxfun(@times, y_te, log_lik_1) + bsxfun(@times,  1-y_te, log_lik_2); %N by S

    %sum_x \log \int  link(x,z) dz
    %max_v = max(tmp,[],2);
    %log_lik = log1p( sum(exp(tmp - max_v),2)-1 ) + max_v - log(size(noise,2));
    %loss.neg_log_lik = -sum(log_lik);

    log_lik =  cmp_log_loss_helper(noise,y_te,X_te);
    loss.neg_log_lik = -sum(log_lik);
end


function log_lik = cmp_log_loss_helper(noise,y_te,X_te)
    y_te = (y_te-0.5)>0;
    y_te = 2*y_te-1;%y_te must be either -1 or 1
    S = size(noise,2);
    N = size(X_te,1);
    maxBatchSize = 50000;
    log_lik = zeros(N,1);
    for i=1:ceil(N/maxBatchSize)
        beginIdx = (i-1)*maxBatchSize + 1;
        endIdx = i*maxBatchSize;
        if endIdx>N
            endIdx =N;
        end
        fn = (X_te(beginIdx:endIdx,:)*noise)';%S by n
        hyp.lik = [];
        lik = {@likLogistic};
        y_hat = repmat(y_te(beginIdx:endIdx)', S, 1);
        f = feval(lik{:}, hyp, y_hat, fn, [], 'infLaplace');
        tmp = f';
        assert(size(tmp,2) == S)
        max_v = max(tmp,[],2);
        log_lik(beginIdx:endIdx,1) = log1p( sum(exp(tmp - max_v),2)-1 ) + max_v - log(size(noise,2));
        %log_lik_check = log(sum(exp(tmp),2)) - log(S)
    end
end
