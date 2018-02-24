function [rn,rp] =Q4c_gradientDescent(X,rn,rp,lambda)
    num_iter = 0;
    while(0 == 0)
        [dF_rn, dF_rp] = Q4c_derivative(X, rn, rp, lambda);
        rn = rn - 1 / ((num_iter + 1) ^ 0.8) * dF_rn;
        rp = rp - 1 / ((num_iter + 1) ^ 0.8) * dF_rp;
        if(num_iter >= 1000000)
            break;
        else
            num_iter = num_iter + 1;
        end
    end
end