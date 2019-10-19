#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:09:20 2019

@author: bhargavjoshi
"""

#Library imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#Tools import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#scaler.fit(self.inputFeatures_train)
#self.inputFeatures_train_scale = scaler.transform(self.inputFeatures_train)
#self.inputFeatures_test_scale = scaler.transform(self.inputFeature_test)

dataset = pd.read_csv("Project3_Dataset_v1.txt",sep=' ', header=None)
DS_array = dataset.to_numpy()
X = DS_array[:,0:2]
y = DS_array[:,2]

clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 5), max_iter = 100000, learning_rate_init = 0.05)
clf.fit(X,y)
predictions = clf.predict(X)
MSE = mean_squared_error(y, predictions)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(X[:,0], X[:,1], y)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(0, 1.0)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(X[:,0], X[:,1], predictions)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(0, 1.0)
plt.show()

print("AMSE: "+str(MSE))