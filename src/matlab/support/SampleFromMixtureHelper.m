% sample from GMM
function [xSampled,logSampDensPerComp] = SampleFromMixtureHelper(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif)

nrSamples = size(rawUnif,1);

% dimensions
k = length(mixMeans{1});
nrComponents = length(mixMeans);

% get cholesky's of the precision matrices of all Gaussian components
cholPrec = cell(nrComponents,1);
for c=1:nrComponents
    cholPrec{c} = chol(mixPrecs{c});
end

% sample mixture component indicators
comps = SampleMixture_comps(logMixWeights, nrSamples, rawUnif);
assert( all(size(rawNorm) == [k, nrSamples]) );

% sample from the Gaussian mixture components
xSampled = zeros(k,nrSamples);
z = rawNorm;

for j=1:nrSamples
    xSampled(:,j) = mixMeans{comps(j)} + cholPrec{comps(j)}\z(:,j);
end

% get densities at the sampled points
logSampDensPerComp = zeros(nrComponents,nrSamples);
for c=1:nrComponents
    if k>1
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*sum((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    else
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    end
end
end

function comps = SampleMixture_comps(llp, nrSamples, rawUnif)
%generate the mixture index according to the log probability weigths (llp)

n=length(llp);
assert(size(llp,1) == 1);
assert(size(llp,2) == n);

if n==1
    comps=ones(nrSamples,1);
else
    assert( all(size(rawUnif) == [nrSamples,1] ) );

    [nllp,ind]=sort(llp,'descend');
    cnllp = zeros(n,1);
    cnllp(1) = nllp(1);
    logsum=nllp(1);
    for i=2:n
        cnllp(i)=log1p(exp(nllp(i)-logsum))+logsum;
        logsum=cnllp(i);
    end
    cpb = (cnllp-logsum)';
    ss=1+sum(repmat(log(rawUnif),1,n)>repmat(cpb,nrSamples,1),2);
    comps = arrayfun(@(x) ind(x), ss);
end
end


