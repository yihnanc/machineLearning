import run_svm as rs
import numpy as np
import os
import csv
from math import log

(X,Y) = rs.get_data("hw4data1.mat")
X = np.delete(X, 0, 0)
Y = np.delete(Y, 0, 0)
C = 4
model = rs.run_svm(X, Y, C)
