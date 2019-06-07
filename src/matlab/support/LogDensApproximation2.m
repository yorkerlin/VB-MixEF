function ld = LogDensApproximation2(x1,x2,mixMean,mixPrec)

k=length(mixMean);
ld=zeros(size(x1));

cholPrec=chol(mixPrec);

for i=1:length(x1)
ld(i) = -(k/2)*log(2*pi)+sum(log(diag(cholPrec))) ...
        -0.5*sum((mixPrec*([x1(i);x2]-mixMean)).*([x1(i);x2]-mixMean));

end
