function [w, xSampled] = get_samples_t(a,z,Mean,Prec)
    %z is a standard/raw gaussian noise (d by s)
    % alpha is a column vector (s by 1)
    assert( size(a,1) == size(z,2) )
    w = invgammaSample(a,a);
    cholPrec = chol(Prec);
    xSampled = Mean + cholPrec\einsum('ij,jk->ijk',z,sqrt(w));
end

