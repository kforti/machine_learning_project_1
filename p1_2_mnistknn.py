#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:31:25 2020

@author: Ryan Ferrin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

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
        dist += (p1[i] -p2[i])**2
    return np.sqrt(dist)


# Find nearest Neighbors
def find_neighbors(train, train_labels, test_p, radius, k):
    distances = []
    for i in range(len(train)-1):
        distance = euclid_distance(test_p, train[i])
        if len(distances) < k or distance < radius:
            distances.append((train_labels[i], distance))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


# Classify test point
def classify(train, train_label, test_p, radius, k):
    neighbors = find_neighbors(train, train_label, test_p, radius, k)
    print(neighbors)
    output_values = [row[-1] for row in neighbors]
    classification = max(set(output_values), key=output_values.count)
    return classification


classification = classify(X_train, y_train, X_test[0], 5, 7)
print(classification)
print(y_test[0])
