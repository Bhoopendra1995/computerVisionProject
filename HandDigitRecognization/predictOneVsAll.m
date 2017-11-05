function p = predictOneVsAll(all_theta, X)


m = size(X, 1);
num_labels = size(all_theta, 1);


p = zeros(size(X, 1), 1);


X = [ones(m, 1) X];

% ====================== CODE HERE ======================


predict = sigmoid(X*all_theta');
[predict_max, predict_i_max]=max(predict, [], 2);
p = predict_i_max;





% =========================================================================


end
