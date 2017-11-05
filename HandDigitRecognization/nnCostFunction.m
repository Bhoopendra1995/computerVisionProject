function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== CODE HERE ======================

yb = zeros(m,num_labels);
syb = size(yb);
for i=1:m
 yb(i,y(i)) = 1;
end 
X1 = [ones(m,1), X];

z1 = X1 * Theta1';
a1 = sigmoid(z1);
sa1 = size(a1);
a11 = [ones(m,1), a1];

z2 = a11 * Theta2';
a2 = sigmoid(z2);
sa2 = size(a2);

% J = 1/m(-y*log(h*x) - (1-y)log(1-h*x)) + (lambda/2m)(theta1^2+ theta2^2)
J = sum(sum( -yb.*log(a2) - (1-yb).*log(1-a2) ,2))/m;

sj = size(sum( -yb.*log(a2) - (1-yb).*log(1-a2) ,2));

% Add the regularization penalty
J = J + (sum(sum((Theta1(:,2:end).^2),2)) + sum(sum((Theta2(:,2:end).^2),2)) )* lambda/(2*m);

sr = size(sum((Theta2(:,2:end).^2),2)) ;

% -------------------------------------------------------------
% Compute gradients using back propagation

% Sample by sample
for t = 1:m
    % Forward to calculate error for sample t
    a_1 = X(t,:)';
    a_1 = [1; a_1];
           
    z_2 = Theta1 * a_1;
    
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    
    z_3 = Theta2 * a_2;
    
    a_3 = sigmoid(z_3);
    
    % Error
    yt = yb(t,:)';
    delta_3 = a_3 - yt; 
    
    % Propagate error backwards
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end); 

    dt2 = delta_3 * a_2';
    dt1 = delta_2 * a_1';
 
    Theta2_grad = Theta2_grad +  delta_3 * a_2';
    Theta1_grad = Theta1_grad + delta_2 * a_1';
end


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad ;

% Add regularization terms
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end