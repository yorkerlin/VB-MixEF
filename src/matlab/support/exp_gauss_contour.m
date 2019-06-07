function cc = exp_gauss_contour(mu, Sigma, alpha)
    k = length(mu);
    assert(k == 2);
    assert(size(mu,1) ==2);
    assert(size(alpha,1) ==2);
    assert( all(size(Sigma) == [2,2]) )
    assert(sum(abs(alpha)) > 0)
    n = 1000;

    mu_x = mu(1);
    std_x = sqrt(Sigma(1)+alpha(1)^2);
    x = linspace(mu_x-10*std_x,mu_x+10*std_x,n);

    mu_y = mu(end);
    std_y = sqrt(Sigma(end)+alpha(end)^2);
    y = linspace(mu_y-10*std_y,mu_y+10*std_y,n);

    [X Y] = meshgrid(x,y); %// all combinations of x, y
    R = [X(:) Y(:)]; % n by k
    L = chol(Sigma,'lower');
    Lalpha = L\alpha;

    alpha_T_Prec_alpha = sum((Lalpha).^2); %alpha'*(Sigma\alpha)
    z = ((R-mu')*(L'\(Lalpha))-1.)/sqrt(alpha_T_Prec_alpha); %n by 1
    log_det_Sigma = 2*sum(log((diag(L)))); %log(det(Sigma))
    tmp = L\(R'-mu); %k by n

    part1 = -( (k-1)*log(2*pi) + log_det_Sigma + log(alpha_T_Prec_alpha) )/2;
    part2 = log_gauss_cdf(z);
    part3 = (z.*z - sum(tmp.*tmp,1)')/2;

    Z = exp(part1+part2+part3); %// compute exp Gaussian pdf
    Z = reshape(Z,size(X));
    cc = contourc(x,y,Z);
end


