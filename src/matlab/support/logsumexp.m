function res=logsumexp(a)
assert( length(a) == length(a(:)) )
if length(a)==1
    res=a;
else
    tmp = a(:);
    max_v=max(tmp);
    res=max_v+log1p( sum( exp(tmp-max_v)  ) - 1. );
end
end
