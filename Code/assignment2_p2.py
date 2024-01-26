# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:47:10 2023

@author: Cristian
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# For classification purposes use RidgeClassifier instead of Ridge
from sklearn.linear_model import RidgeClassifier

# Read a feature space
input_data1 = pd.read_csv("C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged02.csv",header=None)
input_data = input_data1.copy()
NN = 255

# Label/Response set
y = input_data[NN]

# Drop the labels and store the features
input_data.drop(NN,axis=1,inplace=True)
X = input_data

# Generate feature matrix using a Numpy array
tmp = np.array(X)
X1 = tmp[:,0:NN] #tmp[:,0:4]

# Generate label matrix using Numpy array
Y1 = np.array(y)

# Split the data into 80:20
row, col = X.shape
TR = round(row*0.8)
TT = row-TR

# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]

#clf = RidgeClassifier(alpha=1.0)
#ric.fit(X, y)
ric = RidgeClassifier(alpha=0.01)
ric.fit(X1_train, Y1_train)


# Testing with 20% data
X1_test = X1[TR:row,:]
y_test = Y1[TR:row]

#y_test = pd.get_dummies(Y1_test)
yhat_test = ric.predict(X1_test)

# Confusion matrix analytics
CC_test = confusion_matrix(y_test, yhat_test)
#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]
TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print(TN)
print(FP)
print(FN)
print(TP)
print()

# Built-in accuracy measure
from sklearn import metrics
print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))