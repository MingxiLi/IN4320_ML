import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Read data
dataset = pd.read_csv("magic04.csv")

# Replace labels
dataset['Column11'] = dataset['Column11'].map({'g': 0, 'h': 1}).astype(int)

X = dataset.values[:, :-1]
y = dataset.values[:, -1]

# Normalization
X_norm = preprocessing.scale(X)

# Cross-Validation / Supervised LDA
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
acc = 0
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X_norm[train_index], X_norm[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc += accuracy_score(y_test, train_predictions)
acc = acc / 10.0
#print(acc)