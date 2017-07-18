% ########################################################################
% #######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
% ########################################################################

% kmeans_cluster will be used in the experiments, it is available after you 
% have implemented lloyd_iteration and kmeans_obj.


function [best_C, best_a, best_obj] = kmeans_cluster(X, k, init, num_restarts)
  n = size(X,1);
  % Variables for keeping track of the best clustering so far
  best_C = [];
  best_a = [];
  best_obj = Inf;

  for i = 1:num_restarts
    % Choose the initial centers
    if strcmp(init, 'random')
      perm = randperm(n);
      C = X(perm(1:k), :);
    elseif strcmp(init, 'kmeans++')
      C = kmpp_init(X, k);
    elseif strcmp(init, 'fixed')
      C = X(1:k, :);
    end
    % Run the Lloyd iteration until convergence
    [C, a] = lloyd_iteration(X, C);
    % Compute the objective value
    obj = kmeans_obj(X, C, a);
    % If this is the new best objective, then remember C and a:
    if obj < best_obj
      best_C = C;
      best_a = a;
      best_obj = obj;
    end
  end
end
