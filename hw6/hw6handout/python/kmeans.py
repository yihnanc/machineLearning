import numpy as np
import math
import sys

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def update_assignments(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array

  a = np.zeros(X.shape[0])
  #Use for loop to calculate.. slow
  #for i in range(len(X)):
  #  min_dist = sys.float_info.max
  #  for j in range(len(C)):
  #     tmp_dist = np.sqrt(np.sum((X[i]-C[j])**2))
  #     if  (tmp_dist < min_dist):
  #       min_dist = tmp_dist
  #       a[i] = j
  #return a
  for i in range(len(X)):
    cal = np.repeat(X[i][np.newaxis,:], len(C), 0)
    tmp_dist = np.sqrt(np.sum((cal - C)**2, axis = 1))
    a[i] = np.argmin(tmp_dist)
  return a  

def update_centers(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   C is the new cluster centers (k, d), 2-d array
  num = np.zeros(C.shape[0])
  center = np.zeros((C.shape[0], C.shape[1]))
  for i in range(len(a)):
     index = int(a[i])
     center[index] = center[index] + X[i]
     num[index] = num[index] + 1
  # copy the column
  num = np.repeat(num[:,np.newaxis], C.shape[1], 1)
  C = center / num 
  return C



def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array

  a = np.zeros(X.shape[0])
  k = 0
  while 1:
    k = k + 1
    tmp_a = update_assignments(X, C)
    flag_a = 0 
    flag_C = 0
    if np.array_equal(tmp_a, a):
      flag_a = 1
    else:
      a = tmp_a
    tmp_C = update_centers(X, C, a)
    if np.array_equal(tmp_C, C):
      flag_C = 1
    else:
      C = tmp_C
    if flag_a == 1 and flag_C == 1:
      break;
  return (C, a)

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float
  obj = 0.0
  for i in range(len(X)):
    index = int(a[i])
    obj = obj + np.sum((X[i]-C[index])**2)

  return obj


########################################################################
#######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
########################################################################

# kmeans_cluster will be used in the experiments, it is available after you 
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)



########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix
