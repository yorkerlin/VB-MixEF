function [mixWeights,mixMeans,mixPrecs] = mix_gauss_cvi(likelihoodFun,nrComponents,nrSteps,init_m,init_P,callbackFun,dataset_name, preIt, nrSamples, stepSize, decay_mix)

% algorithm settings
if nargin<7
    preIt=floor(nrSteps/50);
    stepSize = 0.0005; %saliman's 2d
    nrSamples = nrComponents*2;
    decay_mix = 0.01;
end

% dimension
k = size(init_m,1);
xMode = init_m;
xNegHess = init_P;

cholNegHess = chol(xNegHess);
mixMeans = cell(nrComponents,1);
mixPrecs = cell(nrComponents,1);

for c=1:nrComponents
    mixMeans{c} = xMode + (cholNegHess\randn(k,1));
    mixPrecs{c} = xNegHess;
end

mixWeights = ones(1,nrComponents)/nrComponents;

%gradient
gmu = cell(nrComponents,1);
gV = cell(nrComponents,1);
for c=1:nrComponents
    gmu{c} = 0*mixMeans{c};
    gV{c} = 0*mixPrecs{c};
end

if nrComponents>1
    tlam_w = log( mixWeights(1,1:end-1) ) - log( mixWeights(1,end) );
    g_mean_m_w = zeros(1,nrComponents-1);
    g_var_m_w = zeros(1,nrComponents-1);
else
    tlam_w = 0;
end
logMixWeights = log(mixWeights);

% do stochastic approximation
for i=1:(nrSteps)
    if nrComponents>1
        max_tlam_w = max(max(tlam_w),0);
        norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
        logMixWeights(1,1:end-1) = tlam_w - norm_log_w;
        logMixWeights(1,end) = -norm_log_w;
    end

    %perform inference

    [zSampled,logSampDensPerComp] = SampleFromMixture(logMixWeights,mixMeans,mixPrecs,nrSamples);
    %zSampled = z ~ q(z) (generate samples from q(z))
    %logSampDensPerComp = log( q(z|w) )

    [logRBindicator,logTotalSampDens] = CombineMixtureComponents(logMixWeights,logSampDensPerComp);
    %logRBindicator = log( q(w|z) )
    %logTotalSampDens =log( q(z) )

    %compute log likelihood with prior  log p(z,x)
    [lpDens,grad,hess] = likelihoodFun(zSampled,i);
    grad = num2cell(grad, [1])';
    hess = squeeze( num2cell(hess,[1,2]) );

    log_sxxWeights = logMixWeights';%log( q(w) )

    %sxyWeights = matrixProd(logRBindicator,(lpDens-logTotalSampDens')/nrSamples);
    %%sxyWeights = E_{q(w|z)} [ log p(z,x) - log q(z)  ]
    [log_sxyWeights sig]= logMatrixProd(logRBindicator,(lpDens-logTotalSampDens')/nrSamples);

    g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig; %Note: q(w|z)/q(w) = q(z|w)/q(z) = \delta (defined at the paper)
    g_m_w2(sig==0) = 0;
    g_m_w2= g_m_w2';

    %the following double loop can be improved by using parallelism
    for c=1:nrComponents
        sH = zeros(k,k);
        sa = zeros(k,1);

        for j=1:nrSamples
            %compute the entropy term
            [grb,hrb] = neg_log_gmm(zSampled(:,j),logRBindicator(:,j),mixMeans,mixPrecs,c);
            % grb = - \nabla_z log q(z)
            % hrb = - \nabla_z^2 log q(z)

            lwt = logRBindicator(c,j) - log_sxxWeights(c);
            sa = sa + weightProd(lwt,grad{j}+grb);
            sH = sH + weightProd(lwt,hess{j}+hrb);
        end
        gmu{c} = sa/nrSamples;
        gV{c} = sH/(2*nrSamples);
    end

    lr1 = stepSize;

    %line search for stepsize
    c = 1;
    while 1
        try
            while c<=nrComponents
                chol( mixPrecs{c} - (2*lr1*(gV{c})) );
                c = c+1;
            end
        catch
            lr1 = lr1*0.5;
            continue;
        end
        break
    end

    for c=1:nrComponents
        mixPrecs{c} = mixPrecs{c} - 2*lr1*(gV{c});
        mixMeans{c} = mixMeans{c} + lr1*(mixPrecs{c}\gmu{c});
    end

    lr2 = lr1*decay_mix;

    %decrease weight rwt at each iteration
    beta1 = 1.0-1e-8;
    decay1 = (1.0-beta1)/(1.0-power(beta1,i));%expoential weighting (adam-like)
    rwt = 5e5*decay1;%set this to zero if the entropy regularization on q(w) is not needed.

    if nrComponents>1
        g_m_w = g_m_w2(1,1:end-1) - g_m_w2(1,end);
        %tlam_w = tlam_w + lr2*g_m_w; %without entropy regularization
        tlam_w = (1.0-rwt*lr2)*tlam_w + lr2*g_m_w; %add an entropy regularization ( rwt * E_q{w}[ -log q(w) ] )
    end

    if nargin>5 && mod(i,preIt) == 0

        if nrComponents>1
            max_tlam_w = max(max(tlam_w),0);
            norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
            mixWeights(1,1:end-1) = exp(tlam_w - norm_log_w);
            mixWeights(1,end) = 1-sum(mixWeights(1,1:end-1));
        end

        post_dist.mixWeights=mixWeights;
        post_dist.mixMeans=mixMeans;
        post_dist.mixPrecs=mixPrecs;

        callbackFun(i,post_dist);

    end
end

if nrComponents>1
    max_tlam_w = max(max(tlam_w),0);
    norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
    logMixWeights(1,1:end-1) = tlam_w - norm_log_w;
    logMixWeights(1,end) = -norm_log_w;
    mixWeights = exp( logMixWeights);
end

end
