loss = [];
r = linspace(-4, 4, 100);
lambda = [0, 1, 2, 3];

for i = 1: 4 
    for j = 1:length(r)
        loss(i, j) = r(j)^2 + 1 + lambda(i) * norm((1 - r(j)));
    end
    subplot(2,2,i);
    plot(r,loss(i,:));
    xlabel('r+');
    ylabel('loss function');
    leg = strcat('lambda = ',num2str(lambda(i)));
    legend(leg);
end