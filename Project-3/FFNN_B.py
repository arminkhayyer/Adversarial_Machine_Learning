#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:09:20 2019

@author: bhargavjoshi
"""
import pandas as pd

dataset = pd.read_csv("Project3_Dataset_v1.txt",sep=' ', header=None, names = ["Feature_1","Feature_2","Output"])
dataset.info()
X = dataset.drop("Output", axis=1)
y = dataset["Output"]

print(X)
print(y)