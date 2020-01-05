# -*- coding: utf-8 -*-
"""
@author: Armin Khayyer
"""
import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
import zipfile
import string
import pandas as pd
from warnings import simplefilter
import os
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from warnings import simplefilter




(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

mask = np.load("mask2.npy")
mask[mask <=200] = 0
print(mask)
plt.imshow(mask)
plt.show()

img = X_train[16]
plt.imshow(img)
plt.show()

img = np.multiply(mask, img)
plt.imshow(img)
plt.show()