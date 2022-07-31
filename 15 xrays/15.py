# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:35:05 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2
import os
from sklearn.model_selection import train_test_split
#%%

directory = 'C:/temp/archive/chest_xray/train/NORMAL/' 
imgs = []

for file in os.listdir(directory):
    img = cv2.imread(directory+file)
    img = cv2.resize(img, (100,100)) # 640, 640 jos riittää muistia
    imgs.append(img)
    
X_normal = imgs[0].reshape(1, imgs[0].shape[0], imgs[0].shape[1], 3)

for img in imgs:
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    X_normal = np.concatenate((X_normal, img), axis = 0)    # X_normal: 1342 x 100 x 100 x 3

#%%
directory = 'C:/temp/archive/chest_xray/test/NORMAL/' 
imgs = []

for file in os.listdir(directory):
    img = cv2.imread(directory+file)
    img = cv2.resize(img, (100,100)) # 640, 640 jos riittää muistia
    imgs.append(img)
    
for img in imgs:
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    X_normal = np.concatenate((X_normal, img), axis = 0)    # X_normal: 1576 x 100 x 100 x 3

#%%

directory = 'C:/temp/archive/chest_xray/train/PNEUMONIA/' 
imgs = []

for file in os.listdir(directory):
    img = cv2.imread(directory+file)
    img = cv2.resize(img, (100,100)) # 640, 640 jos riittää muistia
    imgs.append(img)

X_pneu = imgs[0].reshape(1, imgs[0].shape[0], imgs[0].shape[1], 3)

for img in imgs:
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    X_pneu = np.concatenate((X_pneu, img), axis = 0)    
#%%
 
directory = 'C:/temp/archive/chest_xray/test/PNEUMONIA/' 
imgs = []

for file in os.listdir(directory):
    img = cv2.imread(directory+file)
    img = cv2.resize(img, (100,100)) # 640, 640 jos riittää muistia
    imgs.append(img)
    
for img in imgs:
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    X_pneu = np.concatenate((X_pneu, img), axis = 0)    # X_pneu: 4266 x 100 x 100 x 3
    
#%%

X_normal_shaped = X_normal / 255
X_pneu_shaped = X_pneu / 255

y_normal = np.full(1576, 0) # 0 not sick
y_pneu = np.full(4266, 1) # 1 sick

y = np.concatenate((y_normal, y_pneu), axis = 0)
X = np.concatenate((X_normal_shaped, X_pneu_shaped), axis = 0)
#%%

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)


#%%

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

model = tf.keras.Sequential([
    tf.keras.Input(shape = (100,100,3)),

    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(),
    
    conv_block(32),
    conv_block(64),
    
    conv_block(128),
    tf.keras.layers.Dropout(0.2),
    
    conv_block(256),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    dense_block(512, 0.7),
    dense_block(128, 0.5),
    dense_block(64, 0.3),

    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
#%%
model.compile(loss='binary_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=20, batch_size=100)

#%%  Report the accuracy of your model in both train and test datasets.

train_loss, train_accuracy = model.evaluate(Xtrain)
test_loss, test_accuracy = model.evaluate(Xtest)
print('Accuracy in train data %.3f' % train_accuracy)
print('Accuracy in test data %.3f' % test_accuracy)