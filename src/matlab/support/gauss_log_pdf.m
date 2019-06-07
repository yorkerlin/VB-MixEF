function lp = gauss_log_pdf(x, mu, Prec, alpha)
    k = length(mu);
    assert( size(x,1) == k )
    assert( size(alpha,1) == k )
    Prec_alpha = (Prec*alpha);
    alpha_T_Prec_alpha = alpha'*Prec_alpha;
    inv_S = ( Prec - Prec_alpha*Prec_alpha'/(1.+alpha_T_Prec_alpha) );
    U = chol(inv_S);
    tmp = x-mu;
    lp = -( k*log(2*pi) - 2*sum(log((diag(U)))) + sum(tmp.*(inv_S*tmp),1)  )/2;
end
