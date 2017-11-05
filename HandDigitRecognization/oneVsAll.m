function [all_theta cost_val] = oneVsAll(X, y, num_labels, lambda)


% Some useful variables
m = size(X, 1);
n = size(X, 2);

 
all_theta = zeros(num_labels, n + 1);


X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================



initial_theta = zeros(n + 1, 1);
cost_val = zeros(num_labels,50) ;
for ix = 1:num_labels
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  [theta fx] =  fmincg (@(t)(lrCostFunction(t, X, (y == ix), lambda)), initial_theta, options);
  #costfunction(initial_theta, X, (y == ix), lambda)
  #[theta cosr_history] = gradientDescentMulti(X, y, initial_theta, 0.01, 50)
  cost_val(ix,:) = fx ;
  all_theta(ix,:) = theta;
end


size(cost_val) ;
cost_val(1:5,1:5) ;
% =========================================================================


end
