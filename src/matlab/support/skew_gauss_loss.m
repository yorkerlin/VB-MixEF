function loss = skew_gauss_loss(X_te, y_te, post_dist)
    S = 10000;
    k = size(post_dist.mean,1);
    U = chol(post_dist.preMat);
    w = U\randn(k,S)+post_dist.mean+abs(randn(1,S)).*post_dist.alpha;
    loss = cmp_log_loss(w,y_te,X_te);
end

