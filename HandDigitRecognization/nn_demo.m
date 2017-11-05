

clear ; close all; clc

addpath(genpath('./lib'));

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% ===========Loading and Visualizing Data =============

fprintf('Loading and Visualizing Data ...\n')

load('digit_data2.mat');
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);


displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 8: Training NN ===================

fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 400);

lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

save myfile_th150.mat Theta1
save myfile_th250.mat Theta2

%% ================= Part 9: Visualize Weights =================

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================

pred = predict(Theta1, Theta2, X);
size(pred)
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


