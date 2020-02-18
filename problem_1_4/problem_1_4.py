from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print(X_train[0])
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("MNB Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(confusion_matrix(y_test, y_pred))

clf = LinearDiscriminantAnalysis()
y2_pred = clf.fit(X_train, y_train).predict(X_test)
print("LDA number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y2_pred).sum()))
print(confusion_matrix(y_test, y2_pred))

log = LogisticRegression(random_state=0).fit(X_train, y_train)
y3_pred = log.predict(X_test)
print("LDA number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y3_pred).sum()))
print(confusion_matrix(y_test, y3_pred))

plot_precision_recall_curve(log, X_test, y_test)