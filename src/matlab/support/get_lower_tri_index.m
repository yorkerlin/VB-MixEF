function idx=get_lower_tri_index(mat)
    assert( (size(mat,1)==size(mat,2)) )
    d = size(mat,1);
    idx_mat = tril( reshape(1:(d*d),d,d) );
    idx = idx_mat( idx_mat>0 );
end
