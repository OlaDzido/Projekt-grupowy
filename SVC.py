import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Adult_train.tab", delimiter="\t")
df.drop(index=0, inplace=True)
df.drop(index=1, inplace=True)
for col in df.head(0):
    df[col] = df[col].astype('category')
for col in df.head(0):
    df[col] = df[col].cat.codes
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
df = df.drop(df[to_drop], axis=1)
df = df.sample(n=round(len(df)*0.02))

y = df['y']
X = df.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

#["linear","poly","rbf","sigmoid"]
# Trainig and evaluating the model:
results = {}
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    trainig_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    ### If model has high score save the model
    if test_score >= 0.75:
        with open("SvcAdultModel.pickle", "wb") as f:
            pickle.dump(clf, f)

    results[kernel] = [trainig_score,test_score]


    # Results:
    print(clf.score(X_test,y_test))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

for keys, values in results.items():
    print(keys, ":", values)


def good_model():
    pickle_in = open("SvcAdultModel.pickle", "rb")
    svc = pickle.load(pickle_in)
    return svc
