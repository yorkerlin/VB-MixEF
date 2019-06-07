function [x,w] = hermite_rule(N)
    sigma = 1.;
    kind = 6;
    [x, w] = cgqf (N, kind, 0., 0., 0., 1./(2.*sigma*sigma));
    w = w/(sqrt(2.*pi)*sigma);
end
