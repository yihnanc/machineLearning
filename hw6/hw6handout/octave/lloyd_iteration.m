%########################################################################
%#######  you should maintain the  return type in starter codes   #######
%########################################################################

function [C, a] = lloyd_iteration(X, C)
  % Input:
  %   X is the data matrix (n * d)
  %   C is the initial cluster centers (k * d)
  % Output:
  %   C is the cluster centers (k * d)
  %   a is the cluster assignments (n * 1)

  a = ones(size(X,1), 1);

end
