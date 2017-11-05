function [J, grad] = costfunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


% =============================================================

J1 = sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))) ;

J2 = sum(theta(2:length(theta)).^2);


J=(-1)*J1/m + ((lambda*J2)/(2*m)) ;

end
