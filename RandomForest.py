import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
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


### Split database ###
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

dane = {}



for k in range(25, 75):
    clf = RandomForestClassifier(n_estimators=100,
                               random_state=k,
                               max_features='auto',
                               n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)

    trainig_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test,y_test)
    dane[k] = [trainig_score, test_score]

    if test_score >= 0.85:
        with open("RLadultModel.pickle", "wb") as f:
            pickle.dump(clf, f)



for keys, values in dane.items():
    print(keys, " : ", values)

"""n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in clf.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
train_rf_predictions = clf.predict(X_train)
train_rf_probs = clf.predict_proba(X_train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = clf.predict(X_test)
rf_probs = clf.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt"""
