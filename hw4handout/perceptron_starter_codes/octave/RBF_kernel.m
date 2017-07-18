function K = RBF_kernel(X1,X2, s)
% X1 is a matrix (n * d)
% X2 is a matrix (m * d)
% K is a kernel matrix (n * m)
% s is parameter sigma in RBF function (1 * 1)

%#################################################################################
%####   you should maintain the size of the return value in starter codes  #######
%#################################################################################

  n = size(X1, 1);
  m = size(X2, 1);
  K = zeros(n, m);
end