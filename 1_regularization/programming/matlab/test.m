loss = [];
r = linspace(-4, 4, 100);

    for j = 1:length(r)
        loss(j) = (1-exp(-2*r(j)))/(1+exp(-2*r(j)));
    end
    plot(r,loss);
    xlabel('r+');
    ylabel('loss function');