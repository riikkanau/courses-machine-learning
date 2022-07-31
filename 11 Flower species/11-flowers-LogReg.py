# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:12:03 2021

@author: riikka.naumanen

Use the dataset iris.data to build a classifier model, which predicts the flower species 
from the input variables: Sepal length, Sepal width, Petal length and Petal width

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model


df = pd.read_csv('iris.data', names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

X = np.array(df[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']])

enc = preprocessing.LabelEncoder()
y = np.array(enc.fit_transform(df['Species']))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25)

model = linear_model.LogisticRegression(multi_class = 'multinomial', 
                                        solver = 'newton-cg')
model.fit(Xtrain, ytrain)
trainpredictions = model.predict(Xtrain)
testpredictions = model.predict(Xtest)

# print('Accuracy in train data %.2f' % accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.2f' % accuracy_score(ytest, testpredictions))


