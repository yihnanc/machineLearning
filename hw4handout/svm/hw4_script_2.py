import run_svm as rs
import numpy as np
import os
import csv
from math import log

(X,Y) = rs.get_data("hw4data2.mat")
C = 10
model = rs.run_svm(X, Y, C)
