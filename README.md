Patrycja ***Dąbrowska***
Ola ***Dzido***
Jakub ***Krezel***
Monika ***Sienkiewicz***
Ania ***Walczuk***

#Sprint 1:
Kuba - Oczyszczenie zbioru danych
Ola i Patrycja -podział losoowy zbioru na cz. treningową i cz. walidacyjną
#Sprint 2:
Kuba - SVM
Ola i Patrycja - kNN
Ania i Monika - Decision Tree

----------------------------------------
import pandas as pd
import numpy as np

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