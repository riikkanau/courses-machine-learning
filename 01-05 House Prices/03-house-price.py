# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:31:29 2021

@author: riikka.naumanen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:36:29 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('HousePrice.csv')

# input variable as an numpy array
X = np.array(df[['GrLivArea', 'LotArea', 'YearBuilt']])


# input variable values are scaled to be able to report the relative importance of each input variable
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

# housestyles are categorial (not ordered) and need to be made into dummies 
Xhousestyle = pd.get_dummies(df['HouseStyle'])

# -- Optional: combine very small categories into a single category. Makes the graph values more consistent, less random variance.
for col in Xhousestyle.columns:
    if len(Xhousestyle[Xhousestyle[col] == 1]) < 20:
        Xhousestyle = Xhousestyle.drop(col, axis = 1) 
# --

Xneighborhood = pd.get_dummies(df['Neighborhood'])


# -- Optional: 
# print(Xneighborhood.sum())
# Let's choose 20 as cut off hear too
for col in Xneighborhood.columns:
    if len(Xneighborhood[Xneighborhood[col] == 1]) < 20:
        Xneighborhood = Xneighborhood.drop(col, axis = 1) 
# --

Xscaled = np.concatenate((Xscaled, Xhousestyle, Xneighborhood), axis = 1)

# output variable, which we are trying to predict, as an numpy array
y = np.array(df[['SalePrice']])


# remember to use Xscaled for splitting, if you go with this simple method of scaling whole dataset at once (for scaling train and test sets separately, check video at 7:00)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size = 0.3)


# Building the model, fitting it and doing the predictions based on the fitted model
model = linear_model.LinearRegression()
model.fit(Xtrain, ytrain)
trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)

# After the predictions the scaled versions of the input variables are no longer needed. So let's scale them back.
Xtraincont = scaler.inverse_transform(Xtrain[:,0:3])
Xtrain = np.concatenate((Xtraincont, Xtrain[:,3:]), axis = 1)
Xtestcont = scaler.inverse_transform(Xtest[:,0:3])
Xtest = np.concatenate((Xtestcont, Xtest[:,3:]), axis = 1)

print('Mean absolute error in train data %.2f'
      % mean_absolute_error(ytrain, trainprediction))

print('Mean absolute error in test data %.2f'
      % mean_absolute_error(ytest, testprediction))

print('R2 score in train data %.2f'
      % r2_score(ytrain, trainprediction))

print('R2 score in test data %.2f'
      % r2_score(ytest, testprediction))



plt.figure()
plt.scatter(Xtrain[:,0], ytrain, label = 'Actual prices')
plt.scatter(Xtrain[:,0], trainprediction, label = 'Predicted prices')
plt.title('Train data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

plt.figure()
plt.scatter(Xtest[:,0], ytest, label = 'Actual prices')
plt.scatter(Xtest[:,0], testprediction, label = 'Predicted prices')
plt.title('Test data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()


# Finding out relative importances of the input features and plotting them
feature_importances = model.coef_
features = ['GrLivArea', 'LotArea', 'YearBuilt']
for col in Xhousestyle.columns:
    features.append(col)
    
for col in Xneighborhood.columns:
    features.append(col)
    
plt.figure(figsize = (15,10))
plt.barh(features, feature_importances[0])



