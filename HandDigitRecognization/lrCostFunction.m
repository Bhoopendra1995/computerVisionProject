function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); 


J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

J1 = sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))) ;

J2 = sum(theta(2:length(theta)).^2);


J=(-1)*J1/m + ((lambda*J2)/(2*m)) ;

#J_history(iter) = costfunction(theta ,X, y, lambda);

% =============================================================

grad = sum((sigmoid(X*theta) - y).*X)';
temp = theta; 
temp(1) = 0;   % because we don't add anything for j = 0  
temp = lambda*temp;
grad = grad + temp;
grad = (1/m).*grad;

% =============================================================

grad = grad(:);

end
