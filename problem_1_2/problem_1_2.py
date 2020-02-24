#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  15 11:31:25 2020
WPI CS539 spring 2020
Team Assignment 1 problem 1.2
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time

# load mnist data from sklearn (Geron pg85-86)
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
x_test = X_test_set[:ntest]
y_test = y_test_set[:ntest]

# part 1 Run mnistknndemo
# KNN classifier rewrote mnistknndemo.m from pmtk3

# use the knn model to predict
knnstart = time.time()
k = 5
knn = KNN(n_neighbors=k)
knn.fit(x_train, y_train)
knn_prediction = knn.predict(x_test)
knnstop = time.time()
train_errs = knn.score(x_train, y_train)
test_errs = knn.score(x_test, y_test)
knn_accuracy = accuracy_score(y_test, knn_prediction)
knn_c_time = knnstop - knnstart

print('Accuracy for KNN: ' + str(knn_accuracy * 100))
print('Total execution time for KNN: ' + str(knn_c_time))

# part 2 FLANN

pyflann.set_distance_type('euclidean', order=2)
flann = pyflann.FLANN()
flannstart = time.time()
index_params = flann.build_index(x_train, log_level='info', algorithm='kmeans', branching=32, iterations=7)
neighbor, dist = flann.nn_index(x_test, num_neigbors=5, checks=index_params['checks'])
flann_prediction = y_train[neighbor]
flannstop = time.time()
flann_accuracy = accuracy_score(y_test, flann_prediction)
flann_c_time = flannstop - flannstart

print('Accuracy for Approximate Nearest Neighbors: ' + str(flann_accuracy * 100))
print('Total execution time for Approximate Nearest Neighbors: ' + str(flann_c_time))

# part 3 multi-class logistic regression

logi_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=13500)
logistart = time.time()
logi_reg.fit(x_train, y_train)
logi_prediction = logi_reg.predict(x_test)
logistsop = time.time()
logi_accuracy = accuracy_score(y_test, logi_prediction)
logi_c_time = logistsop - logistart

print('Accuracy for Logistic Regression: ' + str(logi_accuracy * 100))
print('Total execution time for Logistic Regression: ' + str(logi_c_time))
