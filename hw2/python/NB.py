import math
from math import log
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    log_product = float(0)
    for num in x:
        log_product = log_product + num
    return log_product
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float

	#log_product = 0
	#return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, alpha, beta):
    #use map to store the index of each onion/econ
    magazine = []
    #return 2d array
    xmap = np.zeros((2,len(XTrain[0])))
    for i in range(2):
        subMag = []
        magazine.append(subMag)

    #save the index for econ/onion
    for i in range(len(yTrain)):
        #0: econ, 1: onion
        magazine[int(yTrain[i])].append(i)
    #calculate the theta map for onion and econ
    for i in range(2):
        totalCount = float(len(magazine[i]))
        #calculate each word
        for j in range(len(XTrain[0])):
            oneCount = float(0)
            #calculate the number of 1 for each word
            for k in range(len(magazine[i])):
                if XTrain[magazine[i][k]][j] == 1:
                    oneCount = oneCount + 1
            xmap[i][j] = ((oneCount + alpha - 1)/ (totalCount + alpha + beta - 2))
    print(xmap)
    return xmap
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	## Outputs ##
	# D - (2 by V) numpy ndarray

	#D = np.zeros([2, XTrain.shape[1]])
	#return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    p = 0
    for i in range(len(yTrain)):
        if yTrain[i] == 0:
            p = p + 1
    return p / float(len(yTrain))
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float

	#p = 0
	#return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
    #p is prior, D is MAP esitimation
    onion = log(1 - p)
    econ = log(p)
    res = np.zeros(len(XTest))
    for i in range(len(XTest)):
        #0 for all theta of econ, 1 for all theta of onion
        vec = []
        for k in range(2):
            new_sub = []
            vec.append(new_sub)
        for j in range(len(XTest[i])):
            cond_econ = XTest[i][j] * D[0][j] + (1 - XTest[i][j]) * (1 - D[0][j])
            cond_onion = XTest[i][j] * D[1][j] + (1 - XTest[i][j]) * (1 - D[1][j])
            vec[0].append(log(cond_econ))
            vec[1].append(log(cond_onion)) 
        res_econ = logProd(vec[0]) + econ
        res_onion = logProd(vec[1]) + onion
        if (res_econ > res_onion):
            res[i] = float(0)
        else:
            res[i] = float(1)
    return res
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m


	#yHat = np.ones(XTest.shape[0])
	#return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    count = 0
    if len(yHat) != len(yTruth):
        return float(0)
    for i in range(len(yHat)):
        if yHat[i] != yTruth[i]:
            count = count + 1
    return count / float(len(yHat))
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float

	#error = 0
	#return error
