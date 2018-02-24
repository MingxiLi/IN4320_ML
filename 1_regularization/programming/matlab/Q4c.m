clear all;
close all;
clc

data = load('optdigitsubset.txt');
num = length(data);
trn0 = randperm(552,1) + 1;
trn1 = randperm(569,1)+555;
X_trn = [data(trn0, :);data(trn1, :)];
X_test = [data((1:trn0-1), :);data((trn0+1:554), :);data((555:trn1-1), :);data((trn1+1:num), :)];
lambda = [0, 0.1, 1, 10, 100, 1000];
true_error_rate  = [];
apparent_error_rate  = [];
for round = 1:100
for i = 1:length(lambda)
    test_error = 0;
    apparent_error = 0;
    initial_rn = rand(1,64); 
    initial_rp = rand(1,64);
    L_initial = Q4c_lossFunction(initial_rn,initial_rp,X_trn,lambda(i));
    [rn,rp] = Q4c_gradientDescent(X_trn, initial_rn, initial_rp, lambda(i));
    
    %%%%%% true error %%%%%%
    for j = 1:1123
        distance0(j) = sqrt(sum((X_test(j,:)-rn).^2));
        distance1(j) = sqrt(sum((X_test(j,:)-rp).^2));
    end
    for j = 1: 553
        if distance0(j) > distance1(j)
            test_error = test_error + 1;
        end
    end
    for j = 554: 1123
        if distance1(j) > distance0(j)
            test_error = test_error + 1;
        end
    end
    true_error_rate(i,round) = test_error / 1123;
    
    %%%%%% apparent error %%%%%%
    for j = 1:2
        distance_trn0(j) = sqrt(sum((X_trn(j,:)-rn).^2));
        distance_trn1(j) = sqrt(sum((X_trn(j,:)-rp).^2));
    end
    if distance_trn0(1) > distance_trn1(1)
        apparent_error = apparent_error + 1;
    end
    if distance_trn1(2) > distance_trn0(2)
        apparent_error = apparent_error + 1;
    end
    apparent_error_rate(i,round) = apparent_error / 2;
end
end

t_err = sum(true_error_rate,2)/round;
a_err = sum(apparent_error_rate,2)/round;

figure
plot(lambda, t_err, 'b')
hold on
plot(lambda, a_err, 'r')
hold on