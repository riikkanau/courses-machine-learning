# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:39:17 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('Demand.csv', sep=';')

X = np.array(df.iloc[:,:1])
y = np.array(df.iloc[:, 1:])

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)
scalero = preprocessing.StandardScaler()
yscaled = scalero.fit_transform(y)



# Choose days 0 – 250 as your training data 
Xtrain = Xscaled[:250,:]     # Day
ytrain = yscaled[:250,:]     # Demand

# days 251 – 300 as your test data
Xtest = Xscaled[250:,:]      # Day
ytest = yscaled[250:,:]      # Demand


# define what model is used
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation = tf.nn.relu, input_shape = (Xtrain.shape[1],)),
    tf.keras.layers.Dense(30, activation = tf.nn.tanh),
    tf.keras.layers.Dense(30, activation = tf.nn.relu),
    tf.keras.layers.Dense(ytrain.shape[1])
    ])


model.compile(loss = 'mse',
              optimizer = tf.optimizers.Adam(learning_rate = 0.001),
              metrics = ['mae'])


model.fit(Xtrain, ytrain, validation_data = (Xtest, ytest), epochs = 50, batch_size = 10)


# ask prediction from the fitted model, remember to scale back
trainprediction = scalero.inverse_transform(model.predict(Xtrain))
testprediction = scalero.inverse_transform(model.predict(Xtest))

Xtraincont = scaler.inverse_transform(Xtrain)
Xtrain = np.concatenate((Xtraincont, Xtrain), axis = 1)
Xtestcont = scaler.inverse_transform(Xtest)
Xtest = np.concatenate((Xtestcont, Xtest), axis = 1)

ytrain = scalero.inverse_transform(ytrain)
ytest = scalero.inverse_transform(ytest)

# report the accuracy of the model
print('Mean absolute error in train data %.2f'
      % mean_absolute_error(ytrain, trainprediction))

print('Mean absolute error in test data %.2f'
      % mean_absolute_error(ytest, testprediction))

plt.figure()
plt.scatter(Xtrain[:,0], ytrain, label = 'Actual demand')
plt.scatter(Xtrain[:,0], trainprediction, label = 'Predicted demand')
plt.title('Train data prediction')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()

plt.figure()
plt.scatter(Xtest[:,0], ytest, label = 'Actual demand')
plt.scatter(Xtest[:,0], testprediction, label = 'Predicted demand')
plt.title('Test data prediction')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()

# Plot predicted demand along with actual demand for entire data period 0-300 days
days = np.concatenate((Xtrain, Xtest))
demand = np.concatenate((ytrain, ytest))
predicted_demand = np.concatenate((trainprediction, testprediction))

plt.figure()
plt.scatter(days[:,0], demand, label = 'Actual demand')
plt.scatter(days[:,0], predicted_demand, label = 'Predicted demand')
plt.title('Demand during a marketing period')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.legend()