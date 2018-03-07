function[dF_rn,dF_rp] = derivative(X,rn,rp,lambda)
    dF_rn = 2 * rn - 2. * sum(X((1: 554), :)) / 554 + lambda * tanh(rn - rp);
    dF_rp = 2 * rp - 2. * sum(X((555: end), :)) / 571 + lambda * tanh(rp - rn);
end