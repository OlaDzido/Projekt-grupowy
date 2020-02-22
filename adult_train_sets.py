import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

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


n_neighbors = 10


df_original = df
labels = df_original.pop("y")
scaler = StandardScaler()
df = scaler.fit_transform(df_original)
model = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, shuffle=True)
model.fit(X_train, y_train)
a = model.score(X_test, y_test)
print(a)


