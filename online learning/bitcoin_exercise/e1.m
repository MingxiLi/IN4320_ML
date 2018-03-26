clear all
clc

d = 3;
T = 4;
p_a = zeros(d, T);
p_a(:, 1) = [1. / 3., 1. / 3., 1. / 3.]';
p_b = zeros(d, T);
p_b(:, 1) = [1. / 3., 1. / 3., 1. / 3.]';
z = [0, 0, 1, 0; 0.1, 0, 0, 0.9; 0.2, 0.1, 0, 0];
expert_loss = zeros(d, T);

%%%%%% Question A %%%%%%
for t = 2: T
    for i = 1: d
        expert_loss(i, t-1) = z(i, t-1);
    end
    
    % A
    e_c_loss_sum = sum(expert_loss, 2);
    minmum = find(e_c_loss_sum == min(e_c_loss_sum));
    p_a(minmum, t) = 1;
    
    % B
    for i = 1: d
        p_b(i, t) = exp(-e_c_loss_sum(i)) / sum(exp(-e_c_loss_sum));
    end
end
fprintf("p_a:")

for i = 1: d
    expert_loss(i, T) = z(i, T);
end
e_c_loss_sum = sum(expert_loss, 2);

%%%%%% Question b %%%%%%
total_loss_a = zeros(T, 1);
total_loss_b = zeros(T, 1);
for t = 1:T
    if(t == 1)
        total_loss_a(t) = p_a(:, t)' * expert_loss(:, t);
        total_loss_b(t) = p_b(:, t)' * expert_loss(:, t);
    else
        total_loss_a(t) = total_loss_a(t - 1) + p_a(:, t)' * expert_loss(:, t);
        total_loss_b(t) = total_loss_b(t - 1) + p_b(:, t)' * expert_loss(:, t);
    end
end
RnE_a = total_loss_a(T) - min(e_c_loss_sum);
RnE_b = total_loss_b(T) - min(e_c_loss_sum);
%%%%%% Question c %%%%%%
upper_bound = log(d) + min(e_c_loss_sum);


