function log_cdf = log_gauss_cdf(x)
    py.importlib.import_module('scipy');
    assert( size(x,2) == 1 )
    log_cdf = py.scipy.special.log_ndtr( x' );
    log_cdf = double(py.array.array('d',py.numpy.nditer(log_cdf)));
    log_cdf = log_cdf';
    assert( all( size(x) == size(log_cdf) ) )
end

