import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification


def ssclustering(X_l, y_l, X_u):
    X0 = []
    X1 = []
    for index, i in enumerate(X_l):
        if y_l[index] == 0:
            X0.append(i)
        else:
            X1.append(i)
    X0 = np.array(X0)
    X1 = np.array(X1)
    mean0 = np.mean(X0, axis=0)
    mean1 = np.mean(X1, axis=0)
    while(True):
        C0 = X0
        C1 = X1
        for index, i in enumerate(X_u):
            d0 = np.linalg.norm(i - mean0)
            d1 = np.linalg.norm(i - mean1)
            if d0 < d1:
                C0 = np.append(C0, [i], axis=0)
            else:
                C1 = np.append(C1, [i], axis=0)
        mean0n = np.mean(C0, axis=0)
        mean1n = np.mean(C1, axis=0)
        if (mean0n == mean0).all() and (mean1n == mean1).all():
            break
        else:
            mean0 = mean0n
            mean1 = mean1n
    X = np.append(C0, C1, axis=0)
    y = [0 for x in range(0, C0.shape[0])]
    y.extend([1 for xx in range(0, C1.shape[0])])
    y = np.array(y)
    return X, y


def co_training(X_l, y_l, X_u):
    X_labeled = X_l
    y_labeled = y_l
    X_unlabeled = X_u

    while X_unlabeled.shape[0] != 0:
        clf_1 = LogisticRegression()
        clf_1.fit(X_labeled, y_labeled)
        train_predictions_1 = clf_1.predict(X_unlabeled)

        clf_2 = DecisionTreeClassifier()
        clf_2.fit(X_labeled, y_labeled)
        train_predictions_2 = clf_2.predict(X_unlabeled)

        newX_l = []
        newy_l = []
        dele = []
        for i in range(0, len(X_unlabeled)):
            if train_predictions_1[i] == train_predictions_2[i]:
                newX_l.append(X_unlabeled[i])
                newy_l.append(train_predictions_1[i])
                dele.append(i)
        newX_l = np.array(newX_l)
        newy_l = np.array(newy_l)
        if newX_l.shape[0] == 0:
            break
        X_labeled = np.append(X_labeled, newX_l, axis=0)
        y_labeled = np.append(y_labeled, newy_l)
        X_unlabeled = np.delete(X_unlabeled, dele, axis=0)
    return X_labeled, y_labeled


# Read data
X, y = make_classification(n_samples=2000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=2)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)


# Normalization
X_norm = preprocessing.scale(X)

num_labeled = 25
num_unlabeled = [0, 10, 20, 40, 80, 160, 320, 640, 1280]

err_lda = {}
err_clustering = {}
err_co = {}
log_lda = {}
log_clustering = {}
log_co = {}

iteration = range(0, 100)
for n in iteration:
    # Get train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y)
    e_lda = []
    e_clustering = []
    e_co = []
    l_lda = []
    l_clustering = []
    l_co = []

    for index, i in enumerate(num_unlabeled):

        X_labeled = X_train[:num_labeled]
        y_labeled = y_train[:num_labeled]
        X_unlabeled = X_train[num_labeled: num_labeled + num_unlabeled[index]]
        y_unlabeled = y_train[num_labeled: num_labeled + num_unlabeled[index]]

        ###### supervised-LDA ######
        X_trnall = X_train[: num_labeled]
        y_trnall = y_train[: num_labeled]
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_lda.append(1 - accuracy_score(y_test, train_predictions))
        l_lda.append(log_loss(y_test, train_predictions))

        ###### SS-Clustering ######
        if num_unlabeled[index] == 0:
            X_trnall = X_train[: num_labeled]
            y_trnall = y_train[: num_labeled]
        else:
            X_trnall, y_trnall = ssclustering(X_labeled, y_labeled, X_unlabeled)
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_clustering.append(1 - accuracy_score(y_test, train_predictions))
        l_clustering.append(log_loss(y_test, train_predictions))

        ###### co-training ######
        X_trnall, y_trnall = co_training(X_labeled, y_labeled, X_unlabeled)
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_co.append(1 - accuracy_score(y_test, train_predictions))
        l_co.append(log_loss(y_test, train_predictions))

    err_lda[n] = e_lda
    err_clustering[n] = e_clustering
    err_co[n] = e_co
    log_lda[n] = l_lda
    log_clustering[n] = l_clustering
    log_co[n] = l_co

avgerr_lda = []
avgerr_clustering = []
avgerr_co = []
avglog_lda = []
avglog_clustering = []
avglog_co = []

for index, i in enumerate(num_unlabeled):
    sum_lda = 0
    sum_clustering = 0
    sum_co = 0
    sum_log_lda = 0
    sum_log_clustering = 0
    sum_log_co = 0
    for n in err_lda.keys():
        sum_lda = sum_lda + err_lda[n][index]
        sum_clustering = sum_clustering + err_clustering[n][index]
        sum_co = sum_co + err_co[n][index]
        sum_log_lda = sum_log_lda + log_lda[n][index]
        sum_log_clustering = sum_log_clustering + log_clustering[n][index]
        sum_log_co = sum_log_co + log_co[n][index]
    avgerr_lda.append(sum_lda / len(iteration))
    avgerr_clustering.append(sum_clustering / len(iteration))
    avgerr_co.append(sum_co / len(iteration))
    avglog_lda.append(sum_log_lda / len(iteration))
    avglog_clustering.append(sum_log_clustering / len(iteration))
    avglog_co.append(sum_log_co / len(iteration))
print('End')
plt.figure()
plt.plot(num_unlabeled, avgerr_lda, label='lda')
plt.plot(num_unlabeled, avgerr_clustering, label='clustering')
plt.plot(num_unlabeled, avgerr_co, label='co-training')
plt.xlabel('Number of unlabeled samples')
plt.ylabel('Error rate')
plt.legend()

plt.figure()
plt.semilogx(num_unlabeled, avglog_lda, label='lda')
plt.semilogx(num_unlabeled, avglog_clustering, label='clustering')
plt.semilogx(num_unlabeled, avglog_co, label='co-training')
plt.xlabel('Number of unlabeled samples')
plt.ylabel('log-likelihood')
plt.legend()
plt.show()



