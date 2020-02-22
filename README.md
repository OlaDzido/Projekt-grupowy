Patrycja Dąbrowska
Ola Dzido
Jakub Krezel
Monika Sienkiewicz
Ania Walczuk

Sprint 1:
Kuba - Oczyszczenie zbioru danych
Ola i Patrycja -podział losoowy zbioru na cz. treningową i cz. walidacyjną


import pandas as pd


df = pd.read_csv("Adult_train.tab", delimiter="\t")
df.drop(index=0, inplace=True)
df.drop(index=1, inplace=True)
for col in df.head(0):
    df[col] = df[col].astype('category')
for col in df.head(0):
    df[col] = df[col].cat.codes