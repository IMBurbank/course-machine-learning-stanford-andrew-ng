function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% List of possible values for C and sigma
param_vals = [0.01 0.03 0.1 0.3 1 3 10 30];

len = length(param_vals);
error = 1;

for i = 1:len
    for j = 1:len
        temp_C = param_vals(i);
        temp_sigma = param_vals(j);
        
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        pred = svmPredict(model, Xval);
        
        % Classification error
        temp_error = mean(double(pred ~= yval));
        
        if temp_error < error
            error = temp_error;
            C = temp_C;
            sigma = temp_sigma;
        end
    end
end


% =========================================================================

end
