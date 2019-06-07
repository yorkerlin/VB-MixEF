function cc = skew_gauss_contour(mu, Sigma, alpha)
    assert(length(mu) == 2);
    assert(size(mu,1) ==2);
    assert(size(alpha,1) ==2);
    assert( all(size(Sigma) == [2,2]) )
    n = 1000;

    mu_x = mu(1);
    std_x = sqrt(Sigma(1)+alpha(1)^2);
    x = linspace(mu_x-10*std_x,mu_x+10*std_x,n);

    mu_y = mu(end);
    std_y = sqrt(Sigma(end)+alpha(end)^2);
    y = linspace(mu_y-10*std_y,mu_y+10*std_y,n);

    [X Y] = meshgrid(x,y); %// all combinations of x, y
    R = [X(:) Y(:)]; % n by d
    L = chol(Sigma,'lower');
    Lalpha = L\alpha;
    z = (R-mu')*(L'\(Lalpha))/(sqrt(1+sum((Lalpha).^2)));
    %z_check = (R-mu')*(Sigma\alpha)/sqrt(1+alpha'*(Sigma\alpha) );

    %Z = 2*mvnpdf(R,mu',Sigma+alpha*alpha').*normcdf(z); %// compute skew Gaussian pdf
    Z = 2*exp( gauss_log_pdf(R', mu, inv(Sigma), alpha)' ).*normcdf(z); %// compute skew Gaussian pdf
    Z = reshape(Z,size(X));
    %[cc,hh]= contour(X,Y,Z);
    cc = contourc(x,y,Z);
end


