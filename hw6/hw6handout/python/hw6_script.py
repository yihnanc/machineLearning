import os
import csv
import numpy as np
import kmeans
import sys

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')
test = np.zeros((7, 2))
cluster = np.zeros((2, 2))
cluster[0][0] = 1.0
cluster[0][1] = 1.0
cluster[1][0] = 5.0
cluster[1][1] = 7.0
test[0][0] = 1.0
test[0][1] = 1.0
test[1][0] = 1.5
test[1][1] = 2.0
test[2][0] = 3.0
test[2][1] = 4.0
test[3][0] = 5.0
test[3][1] = 7.0
test[4][0] = 3.5
test[4][1] = 5.0
test[5][0] = 4.5
test[5][1] = 5.0
test[6][0] = 3.5
test[6][1] = 4.5
#kmeans.lloyd_iteration(test, cluster)
total = 0.0
for i in range(1000):
  (C, a, obj) = kmeans.kmeans_cluster(X, 9, 'kmeans++', 1)
  total = total + obj
print "ans:"
print total / 1000
result = np.zeros(30)
result[:] = sys.float_info.max
#for i in range(10):
#  (C, a, obj) = kmeans.kmeans_cluster(X, i + 1, 'fixed', 10)
# TODO: Test update_assignments function, defined in kmeans.py

# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py
  #result[i + 1] =  obj

#idx = np.argmin(result)
#print idx
#print result[idx]
# TODO: Test kmeans_obj function, defined in kmeans.py

# TODO: Run experiments outlined in HW6 PDF

# For question 9 and 10
from sklearn.decomposition import PCA
mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
pca = PCA(5)
pca.fit(mnist_X)
reduced = pca.fit_transform(mnist_X)
(C, a, obj) = kmeans.kmeans_cluster(reduced, 3, 'fixed', 1)
print obj
print C
