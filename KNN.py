import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

##################### KNN MODEL ###############################

### Split database ###
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

### Trainig and calibration ###
K = []
trainig = []
test = []
scores = {}

for k in range(2, 21):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    trainig_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test,y_test)


    ### If model has high score save the model
    if test_score >= 0.75:
        with open("KnnAdultModel.pickle", "wb") as f:
            pickle.dump(clf, f)

    K.append(k)
    trainig.append(trainig_score)
    test.append(test_score)
    scores[k] = [trainig_score, test_score]

    """y_pred = clf.predict(X_test)
    print(y_pred)"""

# Evaluating the model:
for keys, values in scores.items():
    print(keys, " : ", values)

# Put results on graph
ax = sns.stripplot(K,trainig)
ax.set(xlabel="values of k", ylabel="Trainig Score")
plt.show()

ax = sns.stripplot(K, test)
ax.set(xlabel="values of k", ylabel="Test Score")

plt.scatter(K, trainig, color='r')
plt.scatter(K, test, color='g')

plt.show()

# Print results:
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# Saved model
def good_model():
    pickle_in = open("KnnAdultModel.pickle", "rb")
    knn = pickle.load(pickle_in)
    return knn