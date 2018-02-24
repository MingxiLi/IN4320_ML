function [rn,rp] =gradientDescent(X,rn,rp,lambda)
    num_iter = 0;
    while(0 == 0)
        [dF_rn, dF_rp] = derivative(X, rn, rp, lambda);
        rn = rn - 1 / (num_iter + 1) ^ 0.8 * dF_rn;
        rp = rp - 1 / (num_iter + 1) ^ 0.8 * dF_rp;
        if(num_iter >= 10000)
            break;
        else
            num_iter = num_iter + 1;
        end
    end
end