# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:42:36 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

df = pd.read_csv('sensor_readings_24.csv')

input_vars = []
for column in range(1,25):
    input_vars.append('Sensor'+str(column))
    
X = np.array(df[input_vars])

enc = preprocessing.LabelEncoder()
y = np.array(enc.fit_transform(df['Command']))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25)


# LogisticRegression is a classification model. It's a binary classifier (0/1) by default, but now we need more categories and need to give extra parameters to do it
model = linear_model.LogisticRegression(multi_class = 'multinomial', 
                                        solver = 'newton-cg')
model.fit(Xtrain, ytrain)
trainpredictions = model.predict(Xtrain)
testpredictions = model.predict(Xtest)

print('Accuracy in train data %.2f' % accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.2f' % accuracy_score(ytest, testpredictions))

# This not very good accuracy yet, lot of walls hit!