# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:36:29 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf

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

Xscaled = np.concatenate((Xscaled, Xhousestyle), axis = 1)

# output variable, which we are trying to predict, as an numpy array
y = np.array(df[['SalePrice']])


scalero = preprocessing.StandardScaler()
yscaled = scalero.fit_transform(y) # this is only using distribution of y -> Law of large numbers -> distribution of y is same as yscaled


Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, yscaled, test_size = 0.3)


# Building the model, fitting it and doing the predictions based on the fitted model
# sequential type of neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation = tf.nn.relu, input_shape = (Xtrain.shape[1],)), # first hidden layer has 30 neurons, activation function is rectified linear function, shape of input layer and how many layers it has
    tf.keras.layers.Dropout(0.3), # fighting overfitting with regularization by adding dropout layers. They randomly shutdown 30% of the neurons of previous layers. Encourages the network to not to trust some particular feature, as the shutdown neurons vary.
    tf.keras.layers.Dense(30, activation = tf.nn.tanh),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(30, activation = tf.nn.relu),
    tf.keras.layers.Dense(ytrain.shape[1]) # one output variable on the output layer, no activation function
    
    ])

# training of the neural network: minimize the loss function by moving the weights of the neural network
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer TF2
model.compile(loss = 'mse', # how far are the predicted house prices from the actual house prices. For regression problem we are choosing mean squared error -loss function. 
              optimizer = tf.optimizers.Adam(learning_rate = 0.001), # Adam: closely related to gradient decent -algorithm, where we are moving the weights of the neural network to the direction of negative gradient. Learning_rate = how much do we move the weights towards it / iteration
              metrics = ['mae']) # we will observe the accuracy of our model on console with mean absolute error

model.fit(Xtrain, ytrain, validation_data = (Xtest, ytest), epochs = 50, batch_size = 10)  # the training date is gone through 50 times and weights are moved every 10 rows of data


#%%

trainprediction = scalero.inverse_transform(model.predict(Xtrain))
testprediction = scalero.inverse_transform(model.predict(Xtest))

# After the predictions the scaled versions of the input variables are no longer needed. So let's scale them back.
Xtraincont = scaler.inverse_transform(Xtrain[:,0:3])
Xtrain = np.concatenate((Xtraincont, Xtrain[:,3:]), axis = 1)
Xtestcont = scaler.inverse_transform(Xtest[:,0:3])
Xtest = np.concatenate((Xtestcont, Xtest[:,3:]), axis = 1)

ytrain = scalero.inverse_transform(ytrain)
ytest = scalero.inverse_transform(ytest)

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



