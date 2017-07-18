import os
import csv
import numpy as np
import NB
from math import log

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py
# TODO: Test NB_XGivenY function, defined in NB.py
#print(XTrainSmall)
tt = []
for i in range(6):
    new = []
    tt.append(new)

tt[0] = [1,0,1,1]
tt[1] = [1,1,0,1]
tt[2] = [0,0,1,0]
tt[3] = [0,1,1,0]
tt[4] = [0,1,0,0]
tt[5] = [1,1,1,1]
print(len(tt[0]))
yy = [0,1,0,1,0,1]
#res = MAP = NB.NB_XGivenY(tt, yy, 2, 5)
MAP = NB.NB_XGivenY(XTrain, yTrain, 2, 5)

# TODO: Test NB_YPrior function, defined in NB.py
yPrior = NB.NB_YPrior(yTrain)

# TODO: Test NB_Classify function, defined in NB.py
res = NB.NB_Classify(MAP, yPrior, XTest)
print(NB.classificationError(res, yTest))
count = 0
for i in range(len(res)):
    if res[i] == float(0):
        count = count + 1
print(count)
print(len(XTest))
# TODO: Test classificationError function, defined in NB.py

# TODO: Run experiments outlined in HW2 PDF
