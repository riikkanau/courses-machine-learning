# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:49:40 2021

@author: rnaum
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('HousePrice.csv')

# input variable as an numpy array
X = np.array(df[['GrLivArea']])

# output variable, which we are trying to predict, as an numpy array
y = np.array(df[['SalePrice']])


# separate 30 % of the data into a test dataset and 70% is left for the training dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3)


# let's build the regression model

# define what model is used
model = linear_model.LinearRegression()

# fit the model parameters to the training data: Xtrain as input, ytrain as output variables
model.fit(Xtrain, ytrain)

# ask prediction from the fitted model
trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)


# report the accuracy of the model
print('Mean absolute error in train data %.2f'
      % mean_absolute_error(ytrain, trainprediction))

print('Mean absolute error in test data %.2f'
      % mean_absolute_error(ytest, testprediction))

print('R2 score in train data %.2f'
      % r2_score(ytrain, trainprediction))

print('R2 score in test data %.2f'
      % r2_score(ytest, testprediction))



plt.figure()
plt.scatter(Xtrain, ytrain, label = 'Actual prices')
plt.scatter(Xtrain, trainprediction, label = 'Predicted prices')
plt.title('Train data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

plt.figure()
plt.scatter(Xtest, ytest, label = 'Actual prices')
plt.scatter(Xtest, testprediction, label = 'Predicted prices')
plt.title('Test data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()