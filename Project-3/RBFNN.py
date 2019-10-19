import numpy as np
import matplotlib.pyplot as plt

import math

def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (np.linalg.norm(x - c)) ** 2)

def rand_cluster(X, k):
    centers = np.random.choice(range(X.shape[0]), size=k)
    print(centers)
    a = []
    for i in centers:
        a.append(X[i])
    return a

def find_closest(centers, x):
    dist = []
    for i, center in enumerate(centers):
        dist.append(np.linalg.norm(center - x))

    return int(np.argmax(np.array(dist)))

def kohonen_unsupervised(X, k):
    centers = []

    for i in range(k):
        centers.append(np.random.uniform(X.min(), X.max(), (1, 2)))

    clusters = [-1] * X.shape[0]
    for num_iter in range(10):
        for j in range(X.shape[0]):
            idx_center = find_closest(centers, X[j, :])
            clusters[j] = idx_center
            centers[idx_center] = centers[idx_center] + 0.5 * (X[j, :] - centers[idx_center])
    sigma = []

    for i in range(k):
        idx = [m for m, e in enumerate(clusters) if e == i]
        dist = (X[idx,:] - centers[i]) ** 2
        dist = np.sum(dist, axis=1)
        sigma.append(np.std(np.sqrt(dist)))

    return centers, sigma

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.01, epochs=10, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kohonen_unsupervised(X, self.k)
        else:
            # use a fixed std
            self.centers = rand_cluster(X, self.k)
            self.centers, _ = kohonen_unsupervised(X, self.k)

            dMax = max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

x_ = np.loadtxt('Project3_Dataset_v1.txt')
train = math.floor(0.7 * x_.shape[0])
eval = math.floor(.8 * x_.shape[0])
test = math.floor(.9 * x_.shape[0])

x_train = x_[:train, 0:2]
y_train = x_[:train, 2]

x_test = x_[train:, 0:2]
y_test = x_[train:, 2]


rbfnet = RBFNet(lr=1e-4, k=100, inferStds=False)
rbfnet.fit(x_train, y_train)

y_pred = rbfnet.predict(x_test)

tp = 0
tn = 0
fp = 0
fn = 0
for test_instance_result, label in zip(y_pred, y_test):
    # print("in calculate", test_instance_result, " ", SchafferF6)
    if ((test_instance_result > 0.5) and (label > 0.5)):
        tp += 1
    if ((test_instance_result <= 0.5) and (label <= 0.5)):
        tn += 1
    if ((test_instance_result > 0.5) and (label <= 0.5)):
        fp += 1
    if ((test_instance_result <= 0.5) and (label > 0.5)):
        fn += 1

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(x_test[:,0], x_test[:,1], y_test)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(-2.0, 2.0)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(x_test[:,0], x_test[:,1], y_pred)
plt.title("Evolved Candidate Solutions")
ax1.set_xlim3d(-100.0, 100.0)
ax1.set_ylim3d(-100.0, 100.0)
ax1.set_zlim3d(-2.0, 2.0)
plt.show()

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn + 0.00001)
precision = tp / (tp + fp + 0.00001)
f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
print("Accuracy:  ", accuracy)
print("Recall:    ", recall)
print("Precision: ", precision)
print("F1:        ", f1)
