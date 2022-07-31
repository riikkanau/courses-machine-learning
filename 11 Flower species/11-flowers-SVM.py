# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:50:08 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = pd.read_csv('iris.data', names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

X = np.array(df[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']])

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

enc = preprocessing.LabelEncoder()
y = np.array(enc.fit_transform(df['Species']))

Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size = 0.25)

model = SVC()
model.fit(Xtrain, ytrain)
trainpredictions = model.predict(Xtrain)
testpredictions = model.predict(Xtest)


# print('Accuracy in train data %.2f' % accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.2f' % accuracy_score(ytest, testpredictions))