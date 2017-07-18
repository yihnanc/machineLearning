import os
import csv
import numpy as np
import perceptron as pp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')
weight = np.zeros(len(XTrain[0]))
#weight[0] = -5
#weight[1] = -2
#tess = np.tile(weight, (5, 1))
#print tess
alpha = np.zeros(len(XTrain))
#after_w = pp.perceptron_train(weight, XTrain, yTrain, 10)
#count = 0
#for i in range(len(yTest)):
#    ypred = pp.perceptron_predict(after_w, XTest[i])
#    if ypred != yTest[i]:
#         count = count + 1
#print count / float(len(yTest))
#pp.perceptron_train(after_w, XTest, yTest, 1)
#print pp.kernel_perceptron_predict(alpha, XTrain, yTrain, XTrain[0], 2)
alpha_after = pp.kernel_perceptron_train(alpha, XTrain, yTrain, 2, 0.1)
count = 0
for i in range(len(XTest)):
    ypred = pp.kernel_perceptron_predict(alpha_after, XTrain, yTrain, XTest[i], 0.1)
    if ypred != yTest[i]:
        count = count + 1
print count / float(len(yTest))
pp.kernel_perceptron_train(alpha_after, XTest, yTest, 1, 1)
#pp.RBF_kernel(XTrain, XTrain, 2)

# Visualize the image
#idx = 0
#datapoint = XTrain[idx, 1:]
#plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
#plt.show()

# TODO: Test perceptron_predict function, defined in perceptron.py

# TODO: Test perceptron_train function, defined in perceptron.py

# TODO: Test RBF_kernel function, defined in perceptron.py

# TODO: Test kernel_perceptron_predict function, defined in perceptron.py

# TODO: Test kernel_perceptron_train function, defined in perceptron.py

# TODO: Run experiments outlined in HW4 PDF
