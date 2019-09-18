function lpd = BetaBinomialLogPosteriorQuad(theta1,theta2,y,n)

lpd=zeros(size(theta1));
for i=1:length(theta1)
        
        eta=exp(theta1(i))/(1+exp(theta1(i)));
        K=exp(theta2);
        
        lpd(i)=sum(betaln(K*eta+y,K*(1-eta)+n-y)-betaln(K*eta,K*(1-eta)));
        
        lpd(i)=lpd(i)+theta2-2*log(1+exp(theta2));
end