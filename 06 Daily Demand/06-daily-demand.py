# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:34:11 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Demand.csv', sep=';')

# Choose days 0 – 250 as your training data 
Xtrain = np.array(df.iloc[:250,:1])     # Day
ytrain = np.array(df.iloc[:250, 1:])    # Demand

# days 251 – 300 as your test data
Xtest = np.array(df.iloc[250:,:1])     # Day
ytest = np.array(df.iloc[250:, 1:])    # Demand

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
plt.scatter(Xtrain, ytrain, label = 'Actual demand')
plt.scatter(Xtrain, trainprediction, label = 'Predicted demand')
plt.title('Train data prediction')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()

plt.figure()
plt.scatter(Xtest, ytest, label = 'Actual demand')
plt.scatter(Xtest, testprediction, label = 'Predicted demand')
plt.title('Test data prediction')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()

# Plot predicted demand along with actual demand for entire data period 0-300 days
days = np.concatenate((Xtrain, Xtest))
demand = np.concatenate((ytrain, ytest))
predicted_demand = np.concatenate((trainprediction, testprediction))

plt.figure()
plt.scatter(days, demand, label = 'Actual demand')
plt.scatter(days, predicted_demand, label = 'Predicted demand')
plt.title('Demand during a marketing period')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()