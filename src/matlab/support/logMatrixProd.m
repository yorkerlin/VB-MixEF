function [res,sig]= logMatrixProd(logw,x)
assert( size(logw,2) == size(x,1) )
res = zeros( size(logw,1) , size(x,2) );
sig = zeros( size(logw,1) , size(x,2) );

for i=1:size(logw,1)
    for j=1:size(x,2)
        sigInd = sign(x(:,j));

        tmp=logw(i,:)'+log(abs(x(:,j)));
        num_post=length(tmp(sigInd>0));
        num_neg=length(tmp(sigInd<0));
        num_zero=length(tmp)-num_neg-num_post;

        if num_post>0 && num_neg>0
            log_positve=logsumexp(tmp(sigInd>0));
            log_negative=logsumexp(tmp(sigInd<0));
            [res(i,j) sig(i,j)] = logsubexp(log_positve,log_negative);
        elseif num_post>0
            res(i,j)= logsumexp(tmp(sigInd>0));
            sig(i,j)= 1;
        elseif num_neg>0
            res(i,j)= logsumexp(tmp(sigInd<0));
            sig(i,j)= -1;
        else
            res(i,j)= -Inf;
            sig(i,j)= 0;
        end
        %if sig(i,j)==0
            %check=0;
        %else
            %check=exp(res(i,j))*sig(i,j);
        %end
        %tmp0=sigInd.*exp(logw(i,:)'+log(abs(x(:,j))));
        %tmp0(sigInd==0) = 0;
        %truth=sum(tmp0);
        %if abs(check -  truth)>1e-11
            %check
            %truth
        %end
        %assert( abs(check -  truth)<1e-11 )
        %disp('ok')
    end
end
end
