#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:31:25 2020

@author: Ryan Ferrin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time

# load mnist data from sklearn (Geron pg85-86)

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist['target']

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]


# Calculate Euclidean distance
def euclid_distance(p1, p2):
    dist = 0
    for i in range(len(p1)-1):
        dist += (p1[i] - p2[i])**2
    return np.sqrt(dist)


# Find nearest Neighbors
def find_neighbors(train, train_labels, test_p, k):
    distances = []
    for i in range(len(train)-1):
        distance = euclid_distance(test_p, train[i])
        if len(distances) < k:
            distances.append((train_labels[i], distance))
            distances.sort(key=lambda tup: tup[1])
        elif distance < distances[k-1][1]:
            distances.append((train_labels[i], distance))
            distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


# Classify test point
def classify(train, train_label, test_p, k):
    neighbors = find_neighbors(train, train_label, test_p, k)
    classification = max(set(neighbors), key=neighbors.count)
    return classification


# KNN
def KNN(train, train_label, test, k):
    predictions=[]
    for test_p in test:
        predictions.append(classify(train, train_label, test_p, k))
    return predictions


def KNN_accuracy(predictions, test):
    correct=0
    for i in range(len(predictions)-1):
        if predictions[i] == test[i]:
            correct += 1
    accuracy = correct/len(predictions)*100
    return accuracy


X_test_t = X_test[:10]
y_test_t = y_test[:10]


runstart = time.time()
predictions =  KNN(X_train, y_train, X_test_t, 7)
runstop = time.time()
accuracy = KNN_accuracy(predictions,y_test_t)

print('Prediction accuracy ' + str(accuracy))
print('TOTAL EXECUTION TIME FOR KNN: ' + str(runstop - runstart))
