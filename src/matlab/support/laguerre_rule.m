function [x,w] = laguerre_rule(N)
    beta = 1.;
    kind = 5;
    [x, w] = cgqf (N, kind, 0., 0., 0., 1./beta);
    w = w/beta;
end
