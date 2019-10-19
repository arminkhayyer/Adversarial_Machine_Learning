import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


x_ = np.loadtxt('Project3_Dataset_v2.txt')
train = math.floor(0.7 * x_.shape[0])
eval = math.floor(.8 * x_.shape[0])
test = math.floor(.9 * x_.shape[0])

x_train = x_[:train, 0:2]
y_train = x_[:train, 2]

x_test = x_[train:, 0:2]
y_test = x_[train:, 2]

kernel='rbf'

if kernel =='linear':
    clf = LinearSVC(random_state=0, tol=1e-4)

if kernel == 'rbf':
    clf = SVC(gamma='auto', kernel='rbf')


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

tp = 0
tn = 0
fp = 0
fn = 0
for test_instance_result, label in zip(y_pred, y_test):

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
