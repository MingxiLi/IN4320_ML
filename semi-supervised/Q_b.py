import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def SSClustering(X_l, y_l, X_u, iteration):
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


# Read data
dataset = pd.read_csv("magic04.csv")
# Replace labels
dataset['Column11'] = dataset['Column11'].map({'g': 0, 'h': 1}).astype(int)

X = dataset.values[:, :-1]
y = dataset.values[:, -1]

# Normalization
X_norm = preprocessing.scale(X)

num_labeled = 25
num_unlabeled = [0, 10, 20, 40, 80, 160, 320, 640]

'''
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_labeled = X_train[:num_labeled]
y_labeled = y_train[:num_labeled]
X_unlabeled = X_train[num_labeled: num_labeled + num_unlabeled]
y_unlabeled = y_train[num_labeled: num_labeled + num_unlabeled]

Xnn, ynn = SSClustering(X_labeled, y_labeled, X_unlabeled, 300)
print(Xnn.shape)
print(ynn.shape)
'''

err_lda = {}
err_clustering = {}
err_cotrn = {}
for n in range(0, 100):
    # Get train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    e_lda = []
    e_clustering = []

    ###### supervised-LDA ######
    for index, i in enumerate(num_unlabeled):
        X_trnall = X_train[: num_labeled + num_unlabeled[index]]
        y_trnall = y_train[: num_labeled + num_unlabeled[index]]
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_lda.append(1 - accuracy_score(y_test, train_predictions))
    err_lda[n] = e_lda

    ###### SS-Clustering ######
    for index, i in enumerate(num_unlabeled):
        if num_unlabeled[index] == 0:
            X_trnall = X_train[: num_labeled]
            y_trnall = y_train[: num_labeled]
        else:
            X_labeled = X_train[:num_labeled]
            y_labeled = y_train[:num_labeled]
            X_unlabeled = X_train[num_labeled: num_labeled + num_unlabeled[index]]
            y_unlabeled = y_train[num_labeled: num_labeled + num_unlabeled[index]]

            X_trnall, y_trnall = SSClustering(X_labeled, y_labeled, X_unlabeled, 10000)
            '''
            for xxxx in range(0, num_unlabeled[index]):
                if y_unlabeled[xxxx] != y_trnall[xxxx]:
                    err = err + 1
            print(err / num_unlabeled[index])
            '''
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_clustering.append(1 - accuracy_score(y_test, train_predictions))
    err_clustering[n] = e_clustering

    ###### co-training ######
    #for index, i in enumerate(num_unlabeled):



avgerr_lda = []
avgerr_clustering = []

for index, i in enumerate(num_unlabeled):
    sum_lda = 0
    sum_clustering = 0
    for n in err_lda.keys():
        sum_lda = sum_lda + err_lda[n][index]
        sum_clustering = sum_clustering + err_clustering[n][index]
    avgerr_lda.append(sum_lda / 100.0)
    avgerr_clustering.append(sum_clustering / 100.0)


plt.plot(num_unlabeled, avgerr_lda, label='lda')
plt.plot(num_unlabeled, avgerr_clustering, label='clustering')
plt.legend()
plt.show()


