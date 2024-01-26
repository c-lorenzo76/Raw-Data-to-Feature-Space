# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:04:38 2023

@author: Cristian
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from sklearn import tree

# Read a feature space
input_data1 = pd.read_csv("C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged12.csv",header=None)

input_data = input_data1.copy()

# Label/Response set
y = input_data[255]

# Drop the labels and store the features
input_data.drop(255,axis=1,inplace=True)
X = input_data

# Generate feature matrix using a Numpy array
X1 = np.array(X)

# Transpose of the feature matrix
X2 = X1.transpose()

# Square of the feature matrix
XX = np.matmul(X2, X1)

# Inverse of the square of the feature matrix
IX = inv(XX)

# Multiply it with its feature matrix
TX = np.matmul(X1, IX)

# Generate label matrix using Numpy array
Y1 = np.array(y)

# Transpose of the label matrix
Y2 = Y1.transpose()

# Multiply it with feature matrix related term
A = np.matmul(Y2, TX)

# Validating the model - compare it with y[1:5]
Z1 = np.matmul(X1[0,:], A)
Z2 = np.matmul(X1[1,:], A)
Z3 = np.matmul(X1[2,:], A)
Z4 = np.matmul(X1[3,:], A)
Z5 = np.matmul(X1[4,:], A)

# Plot the original data
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')
I1 = np.array(input_data)
#ax.scatter3D(I1[:,11], I1[:,32], I1[:,59])
#ax.scatter3D(I1[:,11], I1[:,32], I1[:,59], c=I1[:,59], cmap='Greens')
NN = 100
ax.scatter3D(I1[1:NN,11], I1[1:NN,32], I1[1:NN,59], c=Y1[1:NN])
plt.show()
