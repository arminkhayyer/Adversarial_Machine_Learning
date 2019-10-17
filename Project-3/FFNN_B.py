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
from sklearn.metrics import mean_squared_error, r2_score

class aFFNeuralNetwork:
    def __init__(self,filename):
        self.filename = str(filename)
        self.desiredTarget = []
        self.inputFeatures = []
        
    def ReadDataset(self):
        dataset = pd.read_csv(self.filename,sep=' ', header=None, names = ["Feature_1","Feature_2","Output"])
        dataset.info()
        self.inputFeatures = dataset.drop("Output", axis=1)
        self.desiredTarget = dataset["Output"]
        dataset.describe().transpose()
        
NeuralNetwork1 = aFFNeuralNetwork("Project3_Dataset_v1.txt")
NeuralNetwork1.ReadDataset()