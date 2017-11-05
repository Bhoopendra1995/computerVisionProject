function g = sigmoidGradient(z)


g = zeros(size(z));

% ====================== CODE HERE ======================


g = sigmoid(z).*(1-sigmoid(z));

% =============================================================




end
