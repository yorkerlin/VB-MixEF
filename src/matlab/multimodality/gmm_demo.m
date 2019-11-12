% reproduce the plots in the paper
close all; clear all;
clc

setSeed(1);

% settings
nrSteps=5e4;

% load data
load('./datasets/cancermortality.mat')

y=double(cancermortality.y);
n=double(cancermortality.n);

logp0 = @(theta)BetaBinomialLogPosterior(theta,y,n);

% define log likelihood
logPostDens = @(x)BetaBinomialLogPosterior(x,y,n);
logPostDensQuad = @(x1,x2)BetaBinomialLogPosteriorQuad(x1,x2,y,n);
gradfun = @(x)BetaBinomialGradFun(x,y,n);

% contour grid
[theta1,theta2] = meshgrid(-7.7:.02:-5.6,4.5:0.1:14);
cvals = [-0.15 -0.5 -1 -2 -3 -5];
Z=zeros(size(theta1));

figure()
subplot(3,3,9)
for i=1:size(theta1,1)
    for j=1:size(theta1,2)
        Z(i,j)=logPostDens([theta1(i,j);theta2(i,j)]);
    end
end
mzr = max(max(Z));
normconst = dblquad(@(x1,x2)exp(logPostDensQuad(x1,x2)-mzr),-10,-3.5,2,20);
Z=Z-mzr-log(normconst);
[C,h] = contour(theta1,theta2,Z,cvals);
colormap jet
xlabel('logit u');
ylabel('log P');
title('exact')
true_dist.theta1= theta1;
true_dist.theta2= theta2;
true_dist.Z =Z;


likelihoodFun = @(xSampled,iter)evaluate_likelihood(xSampled,iter,logPostDens,gradfun);

init_m=zeros(2,1);
init_P=eye(2);

nrSamples = 500000;
k = length(init_m);
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);

num_com = 8;
results=cell(num_com,3);

% try multiple mixture components
for i=1:num_com
    disp(['Estimating approximation with ' int2str(i) ' components.'])
    nrComponents=i;

    % fit approximation
    [mixWeights,mixMeans,mixPrecs] = mix_gauss_cvi(likelihoodFun,nrComponents,nrSteps,init_m,init_P);

    % approximation density
    myDens=@(x1,x2)DensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs);
    myLogDens=@(x1,x2)log(myDens(x1,x2));

    % contour plot approximation
    subplot(3,3,i)
    for l=1:size(theta1,1)
        for j=1:size(theta1,2)
            Z(l,j)=myLogDens(theta1(l,j),theta2(l,j));
        end
    end

    [C,h] = contour(theta1,theta2,Z,cvals);
    %[C,h] = contourf(theta1,theta2,Z,10);
    colormap jet
    xlabel('logit u');
    ylabel('log P');
    title(int2str(i))
    results{nrComponents,1} = theta1;
    results{nrComponents,2} = theta2;
    results{nrComponents,3} = Z;
end


function [lp,grad,hess] = evaluate_likelihood(xSampled,iter,logPostFun,gradfun)
%iter is a dummy index in this example
[k,nrSamples] = size(xSampled);
lp = zeros(nrSamples,1);
grad = zeros(k,nrSamples);
if nargout>2
    hess = zeros(k,k,nrSamples);
end

for j=1:nrSamples
    lp(j) = logPostFun(xSampled(:,j));
    if nargout>2
        [grad(:,j),hess(:,:,j)] = gradfun(xSampled(:,j));
    else
        grad(:,j) = gradfun(xSampled(:,j));
    end
end
end

