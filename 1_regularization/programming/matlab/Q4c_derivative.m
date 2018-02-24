function[dF_rn,dF_rp] = Q4c_derivative(X,rn,rp,lambda)
    dF_rn = 2 * rn - 2 * X(1,:) + lambda * sign(rn - rp);
    dF_rp = 2 * rp - 2 * X(2,:) + lambda * sign(rp - rn);
end