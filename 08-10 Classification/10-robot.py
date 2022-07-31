# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:16:39 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

df = pd.read_csv('sensor_readings_24.csv')

input_vars = []
for column in range(1,25):
    input_vars.append('Sensor'+str(column))
    
X = np.array(df[input_vars])

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

ytemp = pd.get_dummies(df['Command']) # retains the order of the control comand columns
y = np.array(ytemp)

Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size = 0.25)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu,
                          input_shape=(Xtrain.shape[1],)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(ytrain.shape[1], activation=tf.nn.softmax) # Softmax-activation function: changes the input data, so that the (in this case four) numbers correspond to probabilities of each category. The sum of them equals one.
    ])

model.compile(loss='categorical_crossentropy', # measures the error between the four output categories
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy']) # accuracy as %

model.fit(Xtrain, ytrain, validation_data=(Xtest,ytest), epochs=50, batch_size=100)

#%%
trainpredictions = np.argmax(model.predict(Xtrain), axis = 1) # max index for each column, so that we can get only one number for each time step
testpredictions = np.argmax(model.predict(Xtest), axis = 1)

#%%

print('Accuracy in train data %.2f' % accuracy_score(np.argmax(ytrain, axis = 1), trainpredictions)) # ytrain info is changed from OneHot-format to one number
print('Accuracy in test data %.2f' % accuracy_score(np.argmax(ytest, axis = 1), testpredictions))

#%%

# maps the predicted numbers with ytemp column headings, ie. the command strings 
mapping = {0:ytemp.columns[0], 1:ytemp.columns[1], 2:ytemp.columns[2], 3:ytemp.columns[3]} 

testpredictionstrings = []
for number in testpredictions:
    testpredictionstrings.append(mapping[number])
    
realcommands = []
for number in np.argmax(ytest, axis = 1):
    realcommands.append(mapping[number])

dfvalidation = pd.DataFrame()
dfvalidation['Prediction'] = testpredictionstrings
dfvalidation['Real Command'] = realcommands

dfsample = dfvalidation.sample(20)
