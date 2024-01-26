# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:27:19 2023

@author: Cristian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Use of PCA
input_data = pd.read_csv("C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged02.csv",header=None)
data = np.array(input_data)
row, col = data.shape
dm = 28 # Try different dimensions
pca1 = PCA(n_components=dm)
score1 = pca1.fit_transform(data[:,0:255]) #col-1
scores1 = pd.DataFrame(score1)
y1 = data[:,255] #Label column
y1d = pd.DataFrame(y1)
sc1 = pd.concat((scores1,y1d),axis=1)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of principal components')
plt.ylabel('cumulative explained variance')
plt.show()
X1 = score1 #tmp[:,0:4]

# Generate label matrix using Numpy array
Y1 = y1

# Machine learning with 80:20
# Split the data into 80:20
row, col = X1.shape
TR = round(row*0.8)
TT = row-TR

# Training with 80% data
X1_train = X1[0:TR,:]
Y1_train = Y1[0:TR]
rF = RandomForestClassifier(random_state=0, n_estimators=500,
oob_score=True, n_jobs=-1)
model = rF.fit(X1_train,Y1_train)
importance = model.feature_importances_
indices = importance.argsort()[::-1]
oob_error = 1 - rF.oob_score_
print(oob_error)

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
print("Our Accuracy Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our Precision Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our Sensitivity Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our Specificity Score:",Specificity)

print('TN = ' + str(TN))
print('FP = ' + str(FP))
print('FN = ' + str(FN))
print('TP = ' + str(TP))
print() 

# Built-in accuracy measure
from sklearn import metrics
print("BuiltIn Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn Precision:",metrics.precision_score(y_test, yhat_test))
print("BuiltIn Sensitivity (recall):",metrics.recall_score(y_test,yhat_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, yhat_test))