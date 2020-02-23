from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, precision_score, \
    roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


from matplotlib import pyplot as plt

def gaussian_classifier(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("MNB Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def linear_discriminant_analysis(X_train, X_test, y_train, y_test):
    clf = LinearDiscriminantAnalysis()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("LDA number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    log = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = log.predict(X_test)
    print("LDA number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def micro_averaged_percision_recall(Y_test, y_score, fpath):

    average_precision = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision))

    precision, recall, _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())

    plt.figure()
    plt.step(recall, precision, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision))
    plt.savefig(fpath)

def micro_averaged_roc(Y_test, y_score, fpath, lw=2):


    precision, recall, _ = roc_curve(Y_test.ravel(), y_score.ravel())
    roc_auc = auc(precision, recall)

    plt.figure()
    plt.plot(precision, recall,
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.close()



def classifier_metrics(test, predictions):
    for pset, name in predictions:
        Y_test = label_binarize(test, classes=[0, 1, 2])
        y_score = label_binarize(pset, classes=[0, 1, 2])

        #micro_averaged_percision_recall(Y_test, y_score, "./problem_1_4/{}_precision_recall_curve".format(name))
        micro_averaged_roc(Y_test, y_score, "./problem_1_4/{}_roc_curve".format(name))


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    gaussian_predictions = gaussian_classifier(X_train, X_test, y_train, y_test)
    lda_predictions = linear_discriminant_analysis(X_train, X_test, y_train, y_test)
    log_reg_predictions = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    predictions = [(gaussian_predictions, "gaussian"),
                   (lda_predictions, "lda"),
                   (log_reg_predictions, "log_reg")]
    classifier_metrics(y_test, predictions)