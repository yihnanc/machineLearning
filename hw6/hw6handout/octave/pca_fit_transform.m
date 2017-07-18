function X_reduced = pca_fit_transform(X, n_components)
% pca_fit_transform projects the data to n_component dimensional space
  mu = mean(X);
  unbiased_X = bsxfun(@minus, X, mu);
  [U, ~] = pca(unbiased_X);
  X_reduced = unbiased_X * U(:, 1:n_components);
end


function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%
  [m, n] = size(X);
  U = zeros(n);
  S = zeros(n);
  Sigma = X' * X/ m;
  [U,S,V] = svd(Sigma);
end
