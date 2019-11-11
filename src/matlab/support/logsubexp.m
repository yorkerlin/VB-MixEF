function [res, sind]=logsubexp(a,b)
assert ( all( size(a) == size(b) ) )
ta=a(:)'; tb=b(:)';
tmp=[ta;tb];
sind=reshape(sign(ta-tb), size(a));

tv=max(tmp,[],1);
res=reshape(tv+log1p(-exp(min(tmp,[],1)-tv)), size(a));

%res=log(abs(exp(a)-exp(b)))
end

