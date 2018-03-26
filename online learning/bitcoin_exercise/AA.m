% Exercise: Aggregating Algorithm (AA)
close all
clear all;
clc
load coin_data;

d = 5;
n = 213;
% strategy matrix: [n, d]
p = zeros(n, d);
% expert cumulative loss matrix: [n, d]
L_expert = zeros(n, d);
% mix loss matrix:[n]
mixLoss = zeros(n, 1);
% expert regret matrix:[n]
Regret = zeros(n, 1);

% compute adversary movez z_t
%%% your code here %%%
z = -log(r);

% compute strategy p_t (see slides)
%%% your code here %%%
for t = 1: n
    if(t == 1)
        L_expert(t, :) = z(t, :);
    else
        L_expert(t, :) = L_expert(t - 1, :) + z(t, :);
    end
end

for t = 1: n
    if(t == 1)
        p(t, :) = [0.2, 0.2, 0.2, 0.2, 0.2];
    else
        p(t, :) = exp(-L_expert(t - 1, :));
        p(t, :) = p(t, :) ./ sum(p(t, :), 2);
    end
end

% compute loss of strategy p_t
%%% your code here %%%
for t = 1: n
    mixLoss(t) = -log(sum(p(t, :) .* exp(-z(t, :)), 2));
end
% compute losses of experts
%%% your code here %%%

% compute regret
%%% your code here %%%
for t = 1: n
    L_c = 0;
    for j = 1: t
        L_c = L_c + mixLoss(j);
    end
    Regret(t) = L_c - min(L_expert(t, :));
end
% compute total gain of investing with strategy p_t
%%% your code here %%%
W = 1;
for t = 1: n
    W = W * p(t, :) * r(t, :)';
end


%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
