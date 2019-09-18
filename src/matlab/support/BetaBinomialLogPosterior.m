function lpd = BetaBinomialLogPosterior(theta,y,n)
eta=exp(theta(1))/(1+exp(theta(1)));
K=exp(theta(2));
N=length(y);

lpd=sum(betaln(K*eta+y,K*(1-eta)+n-y)-betaln(K*eta,K*(1-eta)));

lpd=lpd+theta(2)-2*log(1+exp(theta(2)));