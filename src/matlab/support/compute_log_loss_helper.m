function log_loss = compute_log_loss_helper(y_te, p_hat)
    %y is either 0 or 1
    y_te = (y_te-0.5)>0;
    % log_loss = y.*log2(p) + (1-y).*log2(1-p)
    p_hat = max(eps,p_hat); p_hat = min(1-eps,p_hat);
    err = y_te.*log2(p_hat) + (1-y_te).*log2(1-p_hat);
    log_loss = -mean(err);
end
