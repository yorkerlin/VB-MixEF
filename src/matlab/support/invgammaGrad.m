function [ga, gb] = invgammaGrad(grad,s,alpha,beta)
    ga = grad.*(-(s.*s).*gammaGrad(alpha,beta./s)./beta);
    gb = grad.*(s./beta);
end
