% We compare various approximation of Hessian for VI
clear all;
close all;
setSeed(0);
fs = 14;

% The data is taken from Kevin's book
%% We generate data from two Gaussians:
% x|C=1 ~ gauss([1,5], I)
% x|C=0 ~ gauss([-5,1], 1.1I)
N=30;
D=2;
mu1=[ones(N,1) 5*ones(N,1)];
mu2=[-5*ones(N,1) 1*ones(N,1)];
class1_std = 1;
class2_std = 1.1;
X = [class1_std*randn(N,2)+mu1;2*class2_std*randn(N,2)+mu2];
t = [ones(N,1);zeros(N,1)];
alpha=100; %Variance of prior (alpha=1/lambda)

% Limits and grid size for contour plotting
Range = 20;
Step=0.5;
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

%% prior, likelihood, posterior
f=W*X';
Log_Prior = log(gaussProb(W, zeros(1,D), eye(D).*alpha));
Log_Like = W*X'*t - sum(log(1+exp(f)),2);
Log_Joint = Log_Like + Log_Prior;
post = exp(Log_Joint - logsumexpPMTK(Log_Joint(:),1));


Range = 50;
Step=0.05;
[w11,w22]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n1,n1]=size(w11);
W1=[reshape(w11,n1*n1,1) reshape(w22,n1*n1,1)];


% Identify MAP
[i,j]=max(Log_Joint);
wmap = W(j,:);

% Compute the Laplace Approximation
pp =  preprocessorCreate('addOnes', false, 'standardizeX', false);
model = logregFitBayes(X, t, 'method', 'laplace', 'lambda', 1/alpha, 'preproc', pp);
wMAP = model.wN;
C_Laplace = model.VN;
Laplace_Posterior = gaussProb(W, wMAP', C_Laplace);
Log_Laplace_Posterior = log(Laplace_Posterior + eps);


% Exact VI :
D = 2;
sig = [1;1];
m = zeros(D,1);
v0 = [m; sig(:)];
funObj = @funObj_mfvi_exact;
optMinFunc = struct('display', 0, 'Method', 'lbfgs', 'DerivativeCheck', 'off','LS', 2, 'MaxIter', 100, 'MaxFunEvals', 100, 'TolFun', 1e-4, 'TolX', 1e-4);
gamma = ones(D,1)./alpha;
[v, f, exitflag, inform] = minFunc(funObj, v0, optMinFunc, t, X, gamma);
w_exact_vi = v(1:D);
U = v(D+1:end);
C_exact_vi = diag(U.^2);

% New Algorithms
[N,D] = size(X);
maxIters = 1000;%# of iterations
y = 2*t-1;

nSamples = 20; %# of MC samples
init_P = 100*eye(D) + eye(D)./alpha;
init_m = [1, 1]';

seed = 1;
ss_0 = 0.2; %step size
beta1 = 0; %To enable natural momentum set 0<beta1<1

init_m = [1, 1]';
init_P = 100*eye(D);

M = N;
callbackFun=@(iter,post_dist)plotFun(iter,post_dist,w1,w2,n,post,wmap,w_exact_vi,C_exact_vi,'r'); %plot

priorVar = alpha;
if length(priorVar) ==1
    priorVar = priorVar .* ones(D,1);
end
priorMean = zeros(D,1);

lik_grad =@(samples,iter)LogLogisticLink(samples,iter,X,y,N,M); %logistic likelihood
prior_grad = @(mu,alpha,Prec)prior_gauss_grad(mu,alpha,Prec,priorMean,priorVar); %Gaussian prior

%skew Gaussian fit
skew_cvi_general(lik_grad, prior_grad, callbackFun, nSamples, maxIters, init_m, init_P, ss_0, beta1, seed);


function [gMu_pri, gAlpha_pri, gSigma_pri] = prior_gauss_grad(mu,alpha,Prec,priorMean,priorVar)
  gSigma_pri = - diag(1./priorVar)/2;
  gMu_pri = - (mu-priorMean)./priorVar  - (sqrt(2/pi))*alpha./priorVar;
  gAlpha_pri = - alpha./priorVar - (sqrt(2/pi))*(mu-priorMean)./priorVar;
end


function dummy=plotFun(iter,post_dist,w1,w2,n,post,wmap,w_exact_vi,C_exact_vi,color)
     figure(1);
     cutoff = 1e-2;

     % exact post
     contourf(w1,w2,reshape(post,[n,n]),10, 'linewidth', 1); colorbar;
     hold on
     % map estimate
     plot(wmap(1),wmap(2),'+','color', 'k', 'MarkerSize',4, 'linewidth', 3);

     %% exact VI
     cc = skew_gauss_contour(w_exact_vi, C_exact_vi,[0;0]);%Gaussian contour since alpha=[0;0]
     [xp,yp,zp] = C2xyz(cc);
     hold on;
     once=0;
     plot_leg=[];
     for j = find(zp>cutoff);
         if once==0
             p = plot(xp{j},yp{j},'color', 'b', 'linewidth', 3, 'linestyle', ':','DisplayName','Gauss');
             plot_leg(1)=p;
             once=1;
         else
             plot(xp{j},yp{j},'color', 'b', 'linewidth', 3, 'linestyle', ':');
         end

         hold on;
     end

     % new method
     assert( isfield(post_dist,'alpha') )
     cc = skew_gauss_contour(post_dist.mean, post_dist.covMat,post_dist.alpha);

     [xp,yp,zp] = C2xyz(cc);
     hold on;
     once=0;
     for j = find(zp>cutoff);
         if once==0
             p = plot(xp{j},yp{j},color,'linewidth',3, 'DisplayName','Skew-Gauss');
             plot_leg(2)=p;
             once=1;
         else
             plot(xp{j},yp{j},color,'linewidth',3)
         end
         hold on;
     end
     axis([0 20 0 10]);
     grid on
     legend(plot_leg,'Location','northwest')
     drawnow
end
