function y = sigmoid(x)
    for i = length(x)
        y(i) = 2. / (1 + exp(-x(i))) - 1;
    end
end