# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:51:43 2023

@author: Cristian
"""

import pandas as pd
import numpy as np
#from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Read a feature space
input_data1 = pd.read_csv("C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged02.csv",header=None)
input_data = input_data1.copy()

# Label/Response set
y = input_data[255]

# Drop the labels and store the features
input_data.drop(255,axis=1,inplace=True)
X = input_data

# Generate feature matrix using a Numpy array
tmp = np.array(X)
X1 = tmp[:,0:255] #tmp[:,0:4]

# Generate label matrix using Numpy array
Y1 = np.array(y)

# Machine learning with 80:20
# Split the data into 80:20
row, col = X.shape
TR = round(row*0.8)
TT = row-TR

# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]
rF = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True,
n_jobs=-1)
model = rF.fit(X1_train,Y1_train)
importance = model.feature_importances_
indices = importance.argsort()[::-1]
###################################################################################
#######
#
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)
# for f in range(X.shape[1]):
# print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
# plt.bar(range(X.shape[1]), importance[indices], color="r", yerr=std[indices],align="center")
# plt.xticks(range(X.shape[1]), indices+1, rotation=90)
# plt.show()
###################################################################################
#######
oob_error = 1 - rF.oob_score_

# Testing with 20% data
X1_test = X1[TR:row,:]
y_test = Y1[TR:row]
yhat_test = rF.predict(X1_test)

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
#print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
#print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))
#####################################

### Entropy Measure Testing
import math
# -(7/12)*math.log(7/12, 2) - (5/12)*math.log(5/12, 2)
