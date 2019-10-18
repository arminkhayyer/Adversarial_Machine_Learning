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
import matplotlib.pyplot as pyplt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#Tools import
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

class aFFNeuralNetwork:
    def __init__(self, filename, trainRatio, testRatio):
        self.filename = str(filename)
        self.desiredTarget = []
        self.inputFeatures = []
        self.trainRatio = trainRatio
        self.testRatio = testRatio
        self.desiredTarget_train = []
        self.desiredTarget_test = []
        self.inputFeatures_train = []
        self.inputFeatures_test = []
        self.inputFeatures_train_scale = []
        self.inputFeatures_test_scale = []
        
    def ReadDataset(self):
        dataset = pd.read_csv(self.filename,sep=' ', header=None, names = ["Feature_1","Feature_2","Output"])
        self.inputFeatures = dataset.drop("Output", axis=1)
        self.desiredTarget = dataset["Output"]
        
    def GenerateTrainTestSamples(self):
        inputFeatures_train, inputFeature_test, self.desiredTarget_train, self.desiredTarget_test = train_test_split(self.inputFeatures, 
                                                                                                           self.desiredTarget, 
                                                                                                           train_size = self.trainRatio, 
                                                                                                           test_size = self.testRatio, 
                                                                                                           shuffle = True)
        scaler = StandardScaler()
        scaler.fit(inputFeatures_train)
        self.inputFeatures_train_scale = scaler.transform(inputFeatures_train)
        self.inputFeatures_test_scale = scaler.transform(inputFeature_test) 
        
    def DeployNeuralNetwork(self, iterations, hiddenLayers):
        self.ReadDataset()
        self.GenerateTrainTestSamples()
        mlp = MLPClassifier(hidden_layer_sizes = hiddenLayers, max_iter = iterations)
        mlp.fit(self.inputFeatures_train_scale, self.desiredTarget_train)
        prediction = mlp.predict(self.inputFeatures_test_scale)
        print(confusion_matrix(self.desiredTarget_test,prediction))
        

        
NeuralNetwork1 = aFFNeuralNetwork("Project3_Dataset_v1.txt", trainRatio = 0.7, testRatio = 0.3)
NeuralNetwork1.DeployNeuralNetwork(iterations = 100, hiddenLayers = [10,10,10])