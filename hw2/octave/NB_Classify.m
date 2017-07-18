% The NB_Classify function takes a matrix of MAP estimates for theta_yw,
% the prior probability for class 0, and uses these estimates to classify
% a test set.
function [yHat] = NB_Classify(D, p, XTest)
    %% Inputs %% 
    % D - (2 by V) matrix
    % p - scalar
    % XTest - (m by V) matrix
    
    %% Outputs %%
    % yHat - 1D vector of length m

    yHat = ones(size(XTest,1),1);
end
