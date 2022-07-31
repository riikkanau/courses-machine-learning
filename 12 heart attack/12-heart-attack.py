# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:17:59 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

df = pd.read_csv('heart.csv')

X = np.array(df.iloc[:, :-1]) # get all columns except last and make an array out of them

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

y = np.array(df['output'])

#%%
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size = 0.25)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu,
                          input_shape=(Xtrain.shape[1],)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(units= 2, activation='sigmoid') 
    ])

model.compile(loss='binary_crossentropy', 
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy']) # accuracy as %

model.fit(Xtrain, ytrain, validation_data=(Xtest,ytest), epochs=50)
