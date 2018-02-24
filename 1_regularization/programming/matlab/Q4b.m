clear all;
close all;
clc

data = load('optdigitsubset.txt');
num = length(data);
X = data((1:num),:);
rn_old = zeros(1, 64);
rp_old = zeros(1, 64);
lambda  = [10];
for i = 1:length(lambda)
    initial_rn = rand(1,64); 
    initial_rp = rand(1,64);

    L_initial = lossFunction(initial_rn,initial_rp,X,lambda(i));
    [rn,rp] = gradientDescent(X, initial_rn, initial_rp, lambda(i));
    if(rn == rn_old & rp == rp_old)
        lambda(i)
        break;
    end
    rn_old = rn;
    rp_old = rp;
    
    figure
    img = reshape(rp,[8,8]);
    img = transpose(img);
    img = mat2gray(img);
    subplot(1,2,1)
    imshow(img,'InitialMagnification','fit'); %fit the screen
    leg = strcat('r+ for lambda = ',num2str(lambda(i)));
    title(leg)

    img = reshape(rn,[8,8]);
    img = transpose(img);
    img = mat2gray(img);
    subplot(1,2,2)
    imshow(img,'InitialMagnification','fit'); %fit the screen
    leg = strcat('r- for lambda = ',num2str(lambda(i)));
    title(leg)
end


