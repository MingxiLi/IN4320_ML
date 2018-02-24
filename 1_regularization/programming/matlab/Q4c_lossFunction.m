function[cost] = Q4c_lossFunction(rn,rp,X,lambda)

        cost1 = (X(1,:) - rn).^2;
        cost2 = (X(2,:) - rp).^2;
        cost3 = lambda * abs(rn - rp);
        cost = cost1 + cost2 + cost3;

end