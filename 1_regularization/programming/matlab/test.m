x = -4:0.01:4;
for i = 1:length(x)
    y(i) = (1 - exp(-2*x(i)))/ (1 + exp(-2*x(i)));

end

figure
plot(x, y, 'r')