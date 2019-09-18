function [grad, hess] = BetaBinomialGradFun(theta,y,n)

grad = BetaBinomialGrad(theta,y,n);

if nargout>1
    hess=zeros(2,2);
    hess(:,1) = (BetaBinomialGrad(theta+[1;0]*0.5e-5,y,n)-BetaBinomialGrad(theta-[1;0]*0.5e-5,y,n))/1e-5;
    hess(:,2) = (BetaBinomialGrad(theta+[0;1]*0.5e-5,y,n)-BetaBinomialGrad(theta-[0;1]*0.5e-5,y,n))/1e-5;
    hess=(hess+hess')/2;
end


end

function grad = BetaBinomialGrad(theta,y,n)

eta=exp(theta(1))/(1+exp(theta(1)));
K=exp(theta(2));

gradLpdEta = 0;
gradLpdK = 0;

% gammaln(K*eta+y)
d = sum(psi(K*eta+y));
gradLpdEta = gradLpdEta + K*d;
gradLpdK = gradLpdK + eta*d;

% gammaln(K*(1-eta)+n-y)
d = sum(psi(K*(1-eta)+n-y));
gradLpdEta = gradLpdEta - K*d;
gradLpdK = gradLpdK + (1-eta)*d;

% - gammaln(K+n)
d = sum(psi(K+n));
gradLpdK = gradLpdK - d;

% - gammaln(K*eta)
d = length(y)*psi(K*eta);
gradLpdEta = gradLpdEta - K*d;
gradLpdK = gradLpdK - eta*d;

% - gammaln(K*(1-eta))
d = length(y)*psi(K*(1-eta));
gradLpdEta = gradLpdEta + K*d;
gradLpdK = gradLpdK - (1-eta)*d;

% + gammaln(K)
d = length(y)*psi(K);
gradLpdK = gradLpdK + d;

gradLpdTheta1 = gradLpdEta*(eta-eta^2);
gradLpdTheta2 = gradLpdK*K;

% jacobian
gradLpdTheta2 = gradLpdTheta2 + 1 -2*K/(1+K);

% output
grad = [gradLpdTheta1; gradLpdTheta2];

end
