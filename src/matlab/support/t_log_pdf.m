function [varargout] = t_log_pdf(x, mu, Prec, alpha)
    % v = 2 * alpha  where v is the degree of freedom.
    k = length(mu);
    assert( size(x,1) == k )
    assert( length(alpha) == 1 )
    U = chol(Prec);
    tmp = mu-x;
    Prec_tmp = Prec*tmp;
    common = sum(tmp.*Prec_tmp,1) /(2*alpha);

    %v = 2*alpha;
    %prb = gamma( (v+k)/2 ) / ( gamma(v/2) * power(v*pi,k/2) * power(det( inv(Prec) ), 1/2)  ) * power(  1+ sum((x-mu).*(Prec*(x-mu)), 1 )/v, -(v+k)/2 );
    lp = gammaln(alpha+k/2) - gammaln(alpha) - ( log(2*alpha*pi)*k - 2*sum(log((diag(U)))) )/2 -(alpha+k/2)*log1p(common);

    if nargout>1
        common = 1. + common;
        factor = ( -(alpha+k/2)/(2*alpha) ) ./ common;
        assert( size(factor,2) == size(tmp,2) )
        assert( size(factor,1) == 1 )
        H = einsum('im,jm->ijm', Prec_tmp, Prec_tmp);

        gMu = 2.*(Prec_tmp).*factor;
        gSigma = -Prec/2 + (-H .* reshape(factor,1,1,size(factor,2)));
        assert( size(gSigma,1) == size(gSigma,2) )
        assert( size(gSigma,3) == size(gMu,2) )

        varargout={lp,gMu,gSigma};
    else
        varargout={lp}
    end

end
