% derivatives of negative log GMM
function [grb,hrb] = neg_log_gmm(x,logRBindicator,mixMeans,mixPrecs,c)
%q(z) = GMM(z)

% gradient
gn = zeros(size(x));
for j=1:length(mixMeans)
    %gn = \nabla_z log q(z)
    gn = gn + weightProd(logRBindicator(j),mixPrecs{j}*(mixMeans{j}-x));
end
grb = -gn;

if nargout>1
    % hessian
    hrb = 0*mixPrecs{c};
    for j=1:length(mixMeans)
        %hrb = -\nabla_z^2 log q(z)
        hrb = hrb + weightProd(logRBindicator(j),mixPrecs{j}) - weightProd(logRBindicator(j),(mixPrecs{j}*(mixMeans{j}-x)-gn)*(mixPrecs{j}*(mixMeans{j}-x))');
    end
end

end
