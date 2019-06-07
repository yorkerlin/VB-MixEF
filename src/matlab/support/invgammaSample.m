function [varargout] = invgammaSample(alpha,beta)
    s = beta./gamrnd(alpha, 1.);
    if nargout>1
        %ga = -(1.-s).*(mexGrad(alpha,beta./s)./beta)
        %gb = (1.-s)./(s.*beta)
        mean_grad = (1.-s)./(s.*s);
        [ga, gb] = invgammaGrad(mean_grad, s, alpha, beta);
        varargout = {s, ga, gb};
    else
        varargout = {s};
    end
end


