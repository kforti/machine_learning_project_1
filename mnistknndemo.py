# KNN classifier rewrote mnistknndemo.m from pmtk3 

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time


mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist['target']

ndataset = 60000
X_train_set = X[:ndataset]
y_train_set = y[:ndataset]
X_test_set = X[ndataset:]
y_test_set = y[ndataset:]

ntrain = 10000
ntest = 1000
x_train = X_train_set[:ntrain]
y_train = y_train_set[:ntrain]
x_test = X_test_set[:ntrain]
y_test = y_test_set[:ntrain]


# use the knn model to predict
runstart = time.time()
k = 5
knn = KNN(n_neighbors=k)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
runstop = time.time()
train_errs = knn.score(x_train, y_train)
test_errs = knn.score(x_test, y_test)
accuracy = accuracy_score(y_test, prediction)

print(test_errs)
print(accuracy)
print('TOTAL EXECUTION TIME FOR KNN: ' + str(runstop - runstart))
