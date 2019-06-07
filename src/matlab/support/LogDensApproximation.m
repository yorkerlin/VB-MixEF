function ld = LogDensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs)

% dimensions
nrComponents = length(mixWeights);
k=length(mixMeans{1});

ld=zeros(size(x1));
for i=1:length(x1)

% get densities at x
sampDensPerComp = zeros(nrComponents,1);
for c=1:nrComponents
    cholPrec=chol(mixPrecs{c});
    sampDensPerComp(c) = exp(-(k/2)*log(2*pi)+sum(log(diag(cholPrec))) ...
        -0.5*sum((mixPrecs{c}*([x1(i);x2]-mixMeans{c})).*([x1(i);x2]-mixMeans{c})));
end

RBindicator = sampDensPerComp.*mixWeights';
ld(i) = log(sum(RBindicator));

end
