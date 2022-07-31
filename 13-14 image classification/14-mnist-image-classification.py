# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:57:42 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()

# plt.imshow(Xtest[0], cmap='Greys')

# CNN takes the whole image as an input data: size 28px x 28px, 1 color channel for bw image (would be 3 for color image)
# Also, input variables need to be scaled to be used by neural networks, so let's divide the arrays by maximum number of pixel brightness
Xtrain_shaped = Xtrain.reshape(60000, 28, 28, 1)/255
Xtest_shaped = Xtest.reshape(10000, 28, 28, 1)/255

# for classification problems, the output variable needs to changed to one hot format
ytrainOH = np.array(pd.get_dummies(ytrain))
ytestOH = np.array(pd.get_dummies(ytest))

#%%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, kernel_size=3, activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(30, kernel_size=3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain_shaped, ytrainOH, validation_data=(Xtest_shaped, ytestOH), epochs=20, batch_size=100)

#%%

trainpredictions = np.argmax(model.predict(Xtrain_shaped), axis = 1)
testpredictions = np.argmax(model.predict(Xtest_shaped), axis = 1)

#%%

#plt.imshow(Xtest[96], cmap='Greys')

print('Accuracy in train data %.3f' %accuracy_score(ytrain, trainpredictions))
print('Accuracy in test data %.3f' %accuracy_score(ytest, testpredictions))

#%%

oldwrongpredictions = [43, 104, 115, 247]
for n in oldwrongpredictions:
    prediction = np.argmax(model.predict(Xtest_shaped[n].reshape(1,28,28,1)))
    actual = ytest[n]
    print('Index %.f prediction %.f actual %.f' % (n,prediction, actual))