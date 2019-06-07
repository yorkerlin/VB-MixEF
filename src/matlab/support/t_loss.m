function loss = t_loss(X_te, y_te, post_dist)
    S = 10000;
    k = size(post_dist.mean,1);
    U = chol(post_dist.preMat);

    a=post_dist.alpha*ones(S,1);
    mixw = invgammaSample(a,a);

    w = post_dist.mean + U\einsum('ij,jk->ij',randn(k,S),sqrt(mixw));
    loss = cmp_log_loss(w,y_te,X_te);
end

