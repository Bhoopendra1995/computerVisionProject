


clear ; close all; clc

addpath(genpath('./lib'));
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Loading and Visualizing Data =============


fprintf('Loading and Visualizing Data ...\n')

load('digit_data1.mat'); 
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta cost_val] = oneVsAll(X, y, num_labels, lambda);
fprintf('Program paused. Press enter to continue.\n');
pause;

num_l = size(cost_val,1)
x = 1:1:50 ;




%% ================ Predict for One-Vs-All ================
load('digit_data2.mat'); 
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


 predictOneVsAll(all_theta, X);
