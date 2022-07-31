# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:15 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = pd.read_csv('sensor_readings_24.csv')

input_vars = []
for column in range(1,25):
    input_vars.append('Sensor'+str(column))
    
X = np.array(df[input_vars])

enc = preprocessing.LabelEncoder()
y = np.array(enc.fit_transform(df['Command']))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25)

model = SVC()
model.fit(Xtrain, ytrain)
trainpredictions = model.predict(Xtrain)
testpredictions = model.predict(Xtest)

print('Accuracy in train data %.2f' % accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.2f' % accuracy_score(ytest, testpredictions))

#%%
testpredictionsstrings = enc.inverse_transform(testpredictions)

dfvalidation = pd.DataFrame()
dfvalidation['Prediction'] = testpredictionsstrings
dfvalidation['Real Command'] = enc.inverse_transform(ytest)

dfsample = dfvalidation.sample(20)
