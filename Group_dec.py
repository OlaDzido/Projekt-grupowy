import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import Counter
from statistics import mode
from Exam import predictionsSVM, predictionsRF, predictionsDT, predictionsKNN


# Group decisions
grp_result = defaultdict(list)

# Add decision to dict
for i, x in enumerate(predictionsRF):
    grp_result[i].append(x)

for i, x in enumerate(predictionsDT):
    grp_result[i].append(x)

for i, x in enumerate(predictionsKNN):
    grp_result[i].append(x)

for i, x in enumerate(predictionsSVM):
    grp_result[i].append(x)

d = dict(grp_result)

# Voting
for keys, values in d.items():
    v = mode(values)
    print(keys, " : ", v)
