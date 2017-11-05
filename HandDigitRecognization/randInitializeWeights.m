function W = randInitializeWeights(L_in, L_out)



W = zeros(L_out, 1 + L_in);

% ====================== CODE HERE ======================

INIT_EPSILON = 0.12;
w = rand(L_out, 1 + L_in)*(2*INIT_EPSILON)- INIT_EPSILON;

% =========================================================================

end
