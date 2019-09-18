function [logRBindicator,logTotalSampDens] = CombineMixtureComponents(logMixWeights,logSampDensPerComp)
nrSamples=size(logSampDensPerComp,2);
nrComponents=size(logSampDensPerComp,1);

logRBindicator = logSampDensPerComp + repmat(logMixWeights',1,nrSamples);
max_log_v = max(logRBindicator, [], 1);
logTotalSampDens = log1p(sum( exp(logRBindicator - max_log_v) ,1) -1) + max_log_v;
logRBindicator = logRBindicator - repmat(logTotalSampDens,nrComponents,1);
end

