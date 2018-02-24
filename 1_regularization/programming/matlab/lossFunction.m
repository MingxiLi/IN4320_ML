function[cost] = lossFunction(rn,rp,X,lambda)

        cost1 = sum((X((1:554),:) - rn).^2) / 554;
        cost2 = sum((X((555:end),:) - rp).^2) / 571;
        cost3 = lambda * abs(rn - rp);
        cost = cost1 + cost2 + cost3;

end