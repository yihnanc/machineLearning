import numpy as np
import math
from numpy import linalg as LA

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################


def perceptron_predict(w, x):
  # Input:
  #   w is the weight vector (d,),  1-d array
  #   x is feature values for test example (d,), 1-d array
  # Output:
  #   the predicted label for x, scalar -1 or 1
    z = np.dot(x, np.transpose(w))
    result = 0
    if z <= 0:
        result = -1
    else:
        result = 1
    return result


def perceptron_train(w0, XTrain, yTrain, num_epoch):
  # Input:
  #   w0 is the initial weight vector (d,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  # Output:
  #   the trained weight vector, (d,), 1-d array
    weight = w0
    for i in range(num_epoch):
        #count = 0
        for j in range(len(XTrain)):
            pred_y = perceptron_predict(weight, XTrain[j])  
            if pred_y != yTrain[j]:
	        weight = weight + (yTrain[j] * XTrain[j])       
                #count = count + 1
    #print count
    #print count / float(len(XTrain))
    return weight


def RBF_kernel(X1, X2, sigma):
  # Input:
  #   X1 is a feature matrix (n,d), 2-d array or 1-d array (d,) when n = 1
  #   X2 is a feature matrix (m,d), 2-d array or 1-d array (d,) when m = 1
    #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   K is a kernel matrix (n,m), 2-d array

  #----------------------------------------------------------------------------------------------
  # Special notes: numpy will automatically convert one column/row of a 2d array to 1d array
  #                which is  unexpected in the implementation
  #                make sure you always return a 2-d array even n = 1 or m = 1
  #                your implementation should work when X1, X2 are either 2d array or 1d array
  #                we provide you with some starter codes to make your life easier
  #---------------------------------------------------------------------------------------------
    if len(X1.shape) == 2:
        n = X1.shape[0]
    else:
        n = 1
        X1 = np.reshape(X1, (1, X1.shape[0]))
    if len(X2.shape) == 2:
        m = X2.shape[0]
    else:
        m = 1  
        X2 = np.reshape(X2, (1, X2.shape[0]))
    K = np.zeros((n,m))
    #slower way
    #for i in range(len(X1)):
    #    for j in range(len(X2)):
    #        numerator = (-1.0) * math.pow(LA.norm(X1[i] - X2[j]), 2)
    #        denumerator = 2.0 * math.pow(sigma, 2)
    #        res = np.exp(numerator / denumerator)
    #        K[i][j] = res
    #print K
    for i in range(len(X2)):
        cal = np.tile(X2[i], (len(X1), 1))
        deduct = X1 - cal

        #slowest way, calculate the norm of each row
        #norm = np.zeros(len(deduct))
        #for j in range(len(deduct)):
        #    norm[j] = LA.norm(deduct[j])
        #norm = np.transpose(norm)

        #calculate the norm by element wise multiplication
        norm = np.sum(np.abs(deduct)**2, axis=-1)**(1./2)

        #faster way, but not workable on autolab
        #norm = LA.norm(deduct, axis = 1)        

        result = np.exp(((-1.0) * np.power(norm, 2)) / ((2.0) * np.power(sigma, 2)))
        K[:,i] = result
    #print K.shape 
    return K

def kernel_perceptron_predict(a, XTrain, yTrain, x, sigma):
  # Input:
  #   a is the counting vector (n,),  1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   x is feature values for test example (d,), 1-d array
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the predicted label for x, scalar -1 or 1
    result = RBF_kernel(XTrain, x, sigma)
    ret = 0 
    para = a * yTrain
    para = np.transpose(para)
    para = np.reshape(para, (1, para.shape[0]))
    z = np.dot(para, result) 
    if z <= 0:
        ret = -1
    else:
        ret = 1      
    return ret
 
def kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma):
  # Input:
  #   a0 is the initial counting vector (n,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the trained counting vector, (n,), 1-d array
    for j in range(num_epoch):
        for i in range(len(XTrain)):
            ypred = kernel_perceptron_predict(a0, XTrain, yTrain, XTrain[i], sigma)
            if yTrain[i] != ypred:
                a0[i] = a0[i] + 1
    return a0
