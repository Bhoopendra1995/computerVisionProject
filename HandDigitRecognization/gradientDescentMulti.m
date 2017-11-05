function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)


%num_iters = 2;
m = length(y); 
J_history = zeros(num_iters, 1);
lambda = 1
for iter = 1:num_iters

    % ====================== CODE HERE ======================
      
	grad = sum((sigmoid(X*theta) - y).*X)';
	temp = theta; 
	temp(1) = 0;   % because we don't add anything for j = 0  
	temp = lambda*temp;
	grad = grad + temp;
	grad = (1/m).*grad;

% =============================================================

      grad = grad(:);
      theta = theta - grad  ;
      
    % Saving cost J in every iteration    
    J_history(iter) = costfunction(theta ,X, y, lambda);

end

end
