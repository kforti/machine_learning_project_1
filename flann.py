from pyflann import *
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier as KNN
# from flann_knn import KNeighborsClassifier as FLANN_KNN
from sklearn.datasets import fetch_openml
import numpy as np
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

ntrain = 100
ntest = 10
x_train = X_train_set[:ntrain]
y_train = y_train_set[:ntrain]
x_test = X_test_set[:ntest]
y_test = y_test_set[:ntest]



flann = FLANN()
flannstart = time.time()
k = 5
knn = KNN(n_neighbors=k)
knn.fit(x_train, y_train)
n_outputs = len(knn.classes_)
n_queries = x_test.shape[0]
_y = y_test.reshape((-1, 1))
result, dists = flann.nn(x_train, x_test, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
print(result)
y_pred = np.empty((n_queries, n_outputs)) #, dtype=knn.classes_[0].dtype)
for i, knn.classes_i in enumerate(knn.classes_):
    mode, _ = stats.mode(_y[result, i], axis=1)
    mode = np.asarray(mode.ravel(), dtype=np.intp)
    y_pred[:, i] = classes_i.take(mode)
flannstop = time.time()


print(y_pred)
#print(dists)
print(flannstop -flannstart)
