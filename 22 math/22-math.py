# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:20:43 2021

@author: riikka.naumanen
"""

import numpy as np
import math

# training data
X = np.array([0.05, 0.1])
true = [0.01, 0.99]

# initial values for weights
w1 = np.array([[0.15, 0.25],
               [0.20, 0.30]])

w2 = np.array([[0.40, 0.50],
               [0.45, 0.55]])

# initial values for biases
b = np.array([0, 0])

lr = 0.5 # learning rate
epochs = 10000

def sigmoid(x):
    return 1/(1 + math.exp(-x))

# weighted sum of the values of previous layer
def forwardstep(x, w, bias):
    H = np.matmul(x,w) + bias
    outH = [sigmoid(H[0]), sigmoid(H[1])]
    return outH

def forward(x, w1, w2, b):
    outH = forwardstep(x,w1,b[0])
    outy = forwardstep(outH, w2, b[1])
    return outy
    
def mse(outy, true):
    return 0.5 * np.power((true[0]-outy[0]), 2) +\
        0.5 * np.power((true[1]-outy[1]), 2)
        
def backpropagate(x, w1, w2, b, true, lr):
    outH = forwardstep(x, w1, b[0])
    outy = forwardstep(outH, w2, b[1])
    
    gradw2 = np.zeros((2,2))
    for i in range(0,2):
        for j in range(0,2):
            gradw2[i,j] = (outy[j]-true[j]) * (outy[j]*(1-outy[j])) * outH[i]
    
    gradw1 = np.zeros((2,2))
    for i in range(0,2):
        for j in range(0,2):
            gradw1[i,j] = (outy[0]-true[0]) * (outy[0]*(1-outy[0])) * w2[j][0] *\
                (outH[j] * (1-outH[j])) * X[i] + (outy[1] - true[1]) * (outy[1] * (1-outy[1])) *\
                    w2[j][1] * (outH[j] * (1-outH[j])) * X[i]
    
    bias2 = (outy[0] - true[0]) * outy[0] * (1 - outy[0]) * 1 + (outy[1] - true[1]) * outy[1] * (1 - outy[1]) * 1
    bias1 = (outy[0] - true[0]) * outy[0] * (1 - outy[0]) * w2[0,0] * outH[0] * (1 - outH[0]) * 1 +\
    (outy[1] - true[1]) * outy[1] * (1 - outy[1]) * w2[1,0] * outH[0] * (1 - outH[0]) * 1 +\
    (outy[0] - true[0]) * outy[0] * (1 - outy[0]) * w2[0,1] * outH[1] * (1 - outH[1]) * 1 +\
    (outy[1] - true[1]) * outy[1] * (1 - outy[1]) * w2[1,1] * outH[1] * (1 - outH[1]) * 1
    
    gradb = [lr*bias1, lr*bias2]
                
    neww1 = w1 - lr * gradw1
    neww2 = w2 - lr * gradw2
    newb = b - gradb
    return neww1, neww2, newb



# Run the training
for i in range(0, epochs):
    w1, w2, b = backpropagate(X, w1, w2, b, true, lr)
    E = mse(forward(X, w1, w2, b), true)
    print('Epoch %.i ' % i)
    print('Loss %.6f' % E)

#%%
outy = forward(X, w1, w2, b)
    

        