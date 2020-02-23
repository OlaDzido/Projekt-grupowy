import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


#################### DATA PREPERATION #########################

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


##################### DATA SPLIT ###############################

y = df['y']
X = df.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

##################### KNN MODEL ###############################

pickle_in_knn = open("KnnAdultModel.pickle", "rb")
knn = pickle.load(pickle_in_knn)
knn_pred = knn.predict(X_test)

print("XXXXXXXXXXXXX")
print("k-nearest neighbors algorithm")
print(knn.score(X_test,y_test))
print(confusion_matrix(y_test,knn_pred))
print(classification_report(y_test,knn_pred))

##################### RF MODEL ###############################

pickle_in_rf = open("RFadultModel.pickle", "rb")
rf = pickle.load(pickle_in_rf)
rf_pred = rf.predict(X_test)

print("XXXXXXXXXXXXX")
print("Random Forest algorithm")
print(rf.score(X_test,y_test))
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))


##################### DT MODEL ###############################

pickle_in_dt = open("DTAdultModel.pickle", "rb")
dt = pickle.load(pickle_in_dt)
dt_pred = dt.predict(X_test)

print("XXXXXXXXXXXXX")
print("Decision tree algorithm")
print(dt.score(X_test,y_test))
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))


##################### SVM MODEL ###############################

pickle_in_svm = open("SvcAdultModel.pickle", "rb")
svm = pickle.load(pickle_in_svm)
svm_pred = svm.predict(X_test)

print("XXXXXXXXXXXXX")
print("SVM")
print(dt.score(X_test, y_test))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

#################### MODELS EXAM #########################

"""df_exam = pd.read_csv("Adult_test_without_class.tab", delimiter="\t")
df.drop(index=0, inplace=True)
df.drop(index=1, inplace=True)
for col in df.head(0):
    df[col] = df[col].astype('category')
for col in df.head(0):
    df[col] = df[col].cat.codes
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
df = df.drop(df[to_drop], axis=1)"""

