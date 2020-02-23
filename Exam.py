import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


#### MODELS ####
pickle_in = open("RFadultModel.pickle", "rb")
rf = pickle.load(pickle_in)
pickle_in = open("KnnAdultModel.pickle", "rb")
knn = pickle.load(pickle_in)
pickle_in = open("DTAdultModel.pickle", "rb")
dt = pickle.load(pickle_in)


### EXAM DATA OPENING ###
df_exam = pd.read_csv("Adults_test_without_class.tab", delimiter="\t")
df_exam.drop(index=0, inplace=True)
data = df_exam.drop('y', axis=1)
for col in data.head(0):
    data[col] = data[col].astype('category')
for col in data.head(0):
    data[col] = data[col].cat.codes
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
df = data.drop(data[to_drop], axis=1)

### EXAM ###

predictionsRF = rf.predict(data)
predictionsDT = dt.predict(data)
predictionsKNN = knn.predict(data)


### RESULTS ###
print(f"RF : {predictionsRF}")
print(f"DT : {predictionsDT}")
print(f"KNN : {predictionsKNN}")