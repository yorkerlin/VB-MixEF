function [varargout]= skew_cvi_gernal(lik_grad_fun, prior_grad, callbackFun, nSamples, maxIters, init_m, init_P, ss, beta1, seed)

% initialize
mu = init_m;
Prec = init_P;
chol(Prec); %Prec must be a S.P.D. matrix
D = length(mu);

%initialize alpha
setSeed(seed);
alpha = 0.1*randn(D,1);%since we do not assume any information about alpha, we initialize alpha using a small non-zero value.

%enable natural momentum if beta1>0
assert(beta1 < 1);

old_Prec = Prec;
old_mu = mu;
old_alpha = alpha;

for t = 1:maxIters
  if t == 1 ||  mod(t,floor(maxIters/100)) == 0%
      post_dist.mean = mu;
      post_dist.covMat = Prec\eye(D);
      post_dist.preMat = Prec;
      post_dist.alpha = alpha;

      if mod(t,10) == 0
          callbackFun(t,post_dist);
      end
  end

  Prec_alpha = (Prec*alpha);
  alpha_T_Prec_alpha = alpha'*Prec_alpha;

  setSeed(t*seed)
  gauss_noise = randn(D,nSamples);
  [mix, wt] = get_samples_skew_gauss(alpha,gauss_noise,mu,Prec);%D by s
  wt2 = wt - mix.*alpha;

  %log likelihood
  [~,g_lik,H_lik] = lik_grad(lik_grad_fun,wt,t);
  [~,g2_lik] = lik_grad(lik_grad_fun,wt2,t);

  g = mean(g_lik,2);
  H = mean(H_lik,3);

  factor = (wt-mu)'*Prec_alpha /(1.+ alpha_T_Prec_alpha);
  gAlpha_lik_part1 = factor'.*g_lik; % D by s
  factor2 = sqrt(2/pi)/(1.+ alpha_T_Prec_alpha);
  gAlpha_lik_part2 = factor2*g2_lik;
  gAlpha_lik_part1 = mean(gAlpha_lik_part1,2);
  gAlpha_lik_part2 = mean(gAlpha_lik_part2,2);

  % E_{ N(z|0, alpha_T_Prec_alpha/(alpha_T_Prec_alpha+1)) } [ (z log( 2 cdf(z) )  ] / (alpha_T_Prec_alpha*sqrt(2*pi*(alpha_T_Prec_alpha+1)))
  fun=@(x) expect_log_norm_cdf_grad(x);
  scale = gaussian_integration(0, alpha_T_Prec_alpha/(1.+alpha_T_Prec_alpha), fun ) / (alpha_T_Prec_alpha*sqrt((2*pi)*(1.+alpha_T_Prec_alpha)));

  %Notations:
  %Sigma = inv(Prec);
  %S = Sigma + alpha*alpha';
  inv_S = ( Prec - Prec_alpha*Prec_alpha'/(1.+alpha_T_Prec_alpha) );

  [gMu_pri, gAlpha_pri, gSigma_pri] = prior_grad(mu,alpha,Prec);

  gSigma_lik = H/2;
  gSigma_ent = inv_S/2 - scale*(-Prec_alpha*Prec_alpha');
  gSigma = gSigma_lik + gSigma_pri + gSigma_ent;

  gMu_lik = g;
  gMu_ent = 0;
  gMu = gMu_lik + gMu_pri + gMu_ent;

  gAlpha_lik = gAlpha_lik_part1 + gAlpha_lik_part2;
  gAlpha_ent = inv_S*alpha - scale*(2*Prec_alpha);
  gAlpha = gAlpha_lik + gAlpha_pri + gAlpha_ent;

  decay = (1-beta1)/(1-power(beta1,t));
  momentum = 1-decay;
  lr = ss*decay;

  diff_Prec = Prec - old_Prec;
  diff_alpha = old_Prec*(alpha-old_alpha);
  diff_mu = old_Prec*(mu-old_mu);

  %do a line-search for stepsize
  while 1
      try
          chol( Prec - (2*lr)*gSigma + momentum*diff_Prec );
      catch
          fprintf('line search\n')
          momentum = momentum*0.5;
          lr = lr*0.5;
          continue;
      end
      break
  end

  old_Prec = Prec;
  old_mu = mu;
  old_alpha = alpha;

  % updates
  %Prec_check = Prec - (2*ss)*gSigma
  Prec = Prec - (2*lr)*gSigma + momentum*diff_Prec;

  %mu_check = mu + ss/(1-2/pi)*(Prec_check \ (gMu - sqrt(2/pi)*gAlpha) )
  mu = mu + (Prec\(lr/(1-2/pi)*(gMu - sqrt(2/pi)*gAlpha) + momentum*diff_mu));

  %alpha_check = alpha + ss/(1-2/pi)*(Prec_check \ (gAlpha - sqrt(2/pi)*gMu) )
  alpha = alpha + (Prec\(lr/(1-2/pi)*(gAlpha - sqrt(2/pi)*gMu) + momentum*diff_alpha) );
end

varargout={mu, alpha, Prec};

end


function [varargout] = lik_grad(lik_grad_fun,samples,iter)
if nargout>2
  [lp,grad,Hess] = lik_grad_fun(samples,iter); %log likelihood
  varargout = {lp,grad,Hess};
else
  [lp,grad] = lik_grad_fun(samples,iter); %log likelihood
  varargout = {lp,grad};
end

end

function v = gaussian_integration(mu, sigma2, myfun)
    %1d integration
    [x,w] = gauher(30);
    nx = mu + sqrt(sigma2) * x;
    v = sum( myfun(nx) .* w  );
end

function v = expect_log_norm_cdf_grad(x)
    log_cdf = log_gauss_cdf(x);
    v = x .* ( log(2) + log_cdf );
end
