# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:34:34 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()

# plt.imshow(Xtest[0], cmap='Greys')

# Reshaping the input data to be used for the standard usual neural network: we'll convert each two dimensional array to a single one dimensional vector, ie. each image will be a one dimensional vector (28x28 = 784)
# Also, input variables need to be scaled to be used by neural networks, so let's divide the arrays by maximum number of pixel brightness
Xtrain_flat = Xtrain.reshape(60000, 784)/255
Xtest_flat = Xtest.reshape(10000, 784)/255

# for classification problems, the output variable needs to changed to one hot format
ytrainOH = np.array(pd.get_dummies(ytrain))
ytestOH = np.array(pd.get_dummies(ytest))

#%%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = tf.nn.relu, input_shape = (Xtrain_flat.shape[1],)),
    tf.keras.layers.Dense(50, activation = tf.nn.relu),
    tf.keras.layers.Dense(30, activation = tf.nn.relu),
    tf.keras.layers.Dense(ytrainOH.shape[1], activation = tf.nn.softmax)
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain_flat, ytrainOH, validation_data=(Xtest_flat, ytestOH), epochs=50, batch_size=100)

#%%

trainpredictions = np.argmax(model.predict(Xtrain_flat), axis = 1)
testpredictions = np.argmax(model.predict(Xtest_flat), axis = 1)

#%%

#plt.imshow(Xtest[96], cmap='Greys')

print('Accuracy in train data %.3f' %accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.3f' %accuracy_score(ytest, testpredictions))

#%%

wrongpredictions = []
for i in range(0,len(ytest)):
    if testpredictions[i] != ytest[i]:
        wrongpredictions.append(i)
        
for i in range(0,4):
    plt.figure()
    plt.imshow(Xtest[wrongpredictions[i]], cmap = 'Greys')