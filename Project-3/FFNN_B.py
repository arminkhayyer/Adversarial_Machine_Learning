#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:09:20 2019

@author: bhargavjoshi
"""

#Library imports
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Tools import
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3)
clf = MLPRegressor(solver='sgd', hidden_layer_sizes=[10, 10, 10], max_iter = 50000)
clf.fit(X_train,y_train)
predicted_train = clf.predict(X_train)
predicted_test = clf.predict(X_test)
MSE_train = mean_squared_error(y_train, predicted_train)
MSE = mean_squared_error(y_test, predicted_test)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], y_test)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(0, 1.0)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], predicted_test)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(0, 1.0)
plt.show()

print("\nTraining Error: "+str(MSE_train)+"\nTesting Error: "+str(MSE))
tp = 0
tn = 0
fp = 0
fn = 0

for test_instance_result, label in zip(predicted_test, y_test):

    if ((test_instance_result > 0.5) and (label > 0.5)):
        tp += 1
    if ((test_instance_result <= 0.5) and (label <= 0.5)):
        tn += 1
    if ((test_instance_result > 0.5) and (label <= 0.5)):
        fp += 1
    if ((test_instance_result <= 0.5) and (label > 0.5)):
        fn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn + 0.00001)
precision = tp / (tp + fp + 0.00001)
f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
print("Accuracy:  ", accuracy)
print("Recall:    ", recall)
print("Precision: ", precision)
print("F1:        ", f1)
