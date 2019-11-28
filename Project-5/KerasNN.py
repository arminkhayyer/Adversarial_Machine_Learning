import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
from matplotlib import pyplot as plt
from warnings import simplefilter

# Kernel Setup
simplefilter(action="ignore", category=FutureWarning)
np.random.seed(123)

# Load Data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
plt.imshow(X_train[0])

# Pre-processing data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define Model Architecture
NeuralNetwork = keras.Sequential()
NeuralNetwork.add(keras.layers.Flatten())
NeuralNetwork.add(keras.layers.Dense(128, activation=tf.nn.relu))
NeuralNetwork.add(keras.layers.Dense(128, activation=tf.nn.relu))
NeuralNetwork.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# Train Model
NeuralNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NeuralNetwork.fit(X_train, y_train, epochs=5)

# Test Model
val_loss, val_acc = NeuralNetwork.evaluate(X_test, y_test)
