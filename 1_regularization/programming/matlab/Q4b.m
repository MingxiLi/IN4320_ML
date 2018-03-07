clear all;
close all;
clc

data = load('optdigitsubset.txt');
num = length(data);
X = data((1:num),:);
%img = reshape(X(1,:),[8,8]);
%img = transpose(img);
%img = mat2gray(img);
%imshow(img,'InitialMagnification','fit');
lambda  = [100000];

for i = 1:length(lambda)
    initial_rn = rand(1,64); 
    initial_rp = rand(1,64);

    L_initial = lossFunction(initial_rn,initial_rp,X,lambda(i));
    [rn,rp] = gradientDescent(X, initial_rn, initial_rp, lambda(i));


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

    dif(i) = norm(rn-rp);
end
%{
figure
plot(lambda, dif)
xlabel('lambda')
ylabel('Euclidean distance between r+ and r-')
%}

