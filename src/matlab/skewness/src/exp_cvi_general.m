function [varargout]= exp_cvi_gernal(lik_grad_fun, prior_grad, callbackFun, nSamples, maxIters, init_m, init_P, ss, beta1, seed)

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
  [mix, wt] = get_samples_exp_gauss(alpha,gauss_noise,mu,Prec);%D by s
  wt2 = wt - mix.*alpha;

  %log likelihood
  [~,g_lik,H_lik] = lik_grad(lik_grad_fun,wt,t);
  [~,g2_lik] = lik_grad(lik_grad_fun,wt2,t);

  g = mean(g_lik,2);
  H = mean(H_lik,3);

  factor = ((wt-mu)'*Prec_alpha -1.)/(alpha_T_Prec_alpha);
  gAlpha_lik_part1 = factor'.*g_lik; % D by s
  factor2 = 1./(alpha_T_Prec_alpha);
  gAlpha_lik_part2 = factor2*g2_lik;
  gAlpha_lik_part1 = mean(gAlpha_lik_part1,2);
  gAlpha_lik_part2 = mean(gAlpha_lik_part2,2);

  [gMu_ent, gAlpha_ent, gSigma_ent] = ent_grad(alpha_T_Prec_alpha,Prec_alpha,Prec);
  [gMu_pri, gAlpha_pri, gSigma_pri] = prior_grad(mu,alpha,Prec);

  gSigma_lik = H/2;
  gSigma = gSigma_lik + gSigma_pri + gSigma_ent;

  gMu_lik = g;
  gMu = gMu_lik + gMu_pri + gMu_ent;

  gAlpha_lik = gAlpha_lik_part1 + gAlpha_lik_part2;
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

  %mu_check = mu + ss*(Prec_check \ (2*gMu - gAlpha) )
  mu = mu + (Prec\(lr*(2.*gMu -gAlpha) + momentum*diff_mu));

  %alpha_check = alpha + ss*(Prec_check \ (gAlpha - gMu) )
  alpha = alpha + (Prec\(lr*(gAlpha - gMu) + momentum*diff_alpha) );
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


function [gMu, gAlpha, gSigma] =  ent_grad(alpha_T_Prec_alpha,Prec_alpha,Prec)
    %compute the grad of E_q(z) [ - log q(z) ]
    %Exponentially Modified Gaussian

    y = sqrt(alpha_T_Prec_alpha);

    function v = expect_exp_log_cdf_grad(w,z)
        t = z + w*y - 1./y;
        log_cdf = log_gauss_cdf(t);

        p1 = -exp(-t.*t/2. - log_cdf)/sqrt(2.*pi);

        v = p1 .* (w+ 1./alpha_T_Prec_alpha);
    end

    fun=@(w,z) expect_exp_log_cdf_grad(w,z);
    scale = integration_2d( fun );

    gMu = 0;
    gAlpha = scale*(Prec_alpha)/y + Prec_alpha/alpha_T_Prec_alpha +Prec_alpha/(alpha_T_Prec_alpha*alpha_T_Prec_alpha);
    gSigma = scale*(-Prec_alpha*Prec_alpha')/(2.*y) +( Prec - Prec_alpha*Prec_alpha'/alpha_T_Prec_alpha - Prec_alpha*Prec_alpha'/(alpha_T_Prec_alpha*alpha_T_Prec_alpha)  )/2.;
end

function v = integration_2d(myfun)
    s = 30;
    [w,wt_w] = laguerre_rule(s); %w ~ Exp(1)
    w=repmat(w,1,s)';
    wt_w=repmat(wt_w,1,s)';

    %[z,wt_z] = hermite_rule(s); %z ~ Gauss(0,1)
    [z,wt_z] = gauher(s); %z ~ Gauss(0,1)

    z=repmat(z,1,s);
    wt_z=repmat(wt_z,1,s);


    %2d integration
    v = sum( myfun(w(:),z(:)) .* wt_w(:) .* wt_z(:) );
end

