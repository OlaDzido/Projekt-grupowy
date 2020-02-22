import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics


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

train, test = train_test_split(df, test_size=0.2, shuffle=True)

train.to_csv("train.csv", sep = ";", encoding = "utf-8", index = False, header = False)
test.to_csv("train.csv", sep = ";", encoding = "utf-8", index = False, header = False)

