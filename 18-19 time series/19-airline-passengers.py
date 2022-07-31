# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:19:54 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

df = pd.read_csv('AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df['Time'] = df.index

seqlength = 12 # length of the repeating time sequence in months
predictiontime = 12 # months forward we are going to predict

# Create a stationary time series (ie. moving the trend information)
df['PassengerLag'] = df['Passengers'].shift(1)
df['PassengerDiff'] = df.apply(lambda row: 
                           row['Passengers'] - row['PassengerLag'], axis = 1)

# history of number of the passengers for the previous 12 months. They'll become the input variables.
for i in range(1, seqlength):
    df['PassengerDiffLag' + str(i)] = df['PassengerDiff'].shift(i)

# future numbers of the passengers for the next 12 months. They'll become the output variables.    
for i in range(1, predictiontime + 1):
    df['PassengerDiffFut' + str(i)] = df['PassengerDiff'].shift(-i)


# separating the test data. As each row contains also info about previous 12 months, we need test data from 1967 and 1968, not just the latter.
df_train = df.iloc[:-2 * predictiontime]
df_train.dropna(inplace = True)
df_test = df.iloc[-2 * predictiontime:]

# define and scale variables
input_vars = ['PassengerDiff']
for i in range(1, seqlength):
    input_vars.append('PassengerDiffLag' + str(i))

output_vars = []
for i in range(1, predictiontime + 1):
    output_vars.append('PassengerDiffFut' + str(i))
    
scaler = preprocessing.StandardScaler()
scalero = preprocessing.StandardScaler()

X = np.array(df_train[input_vars])
X_scaled = scaler.fit_transform(X)
X_scaledLSTM = X_scaled.reshape(X.shape[0], X.shape[1], 1) # 2D input data is reshaped to 3D. The last number 1 stands for number of separate time series, we are using for input. 
y = np.array(df_train[output_vars])
y_scaled = scalero.fit_transform(y)

X_test = np.array(df_test[input_vars])
X_testscaled = scaler.transform(X_test)
X_testscaledLSTM = X_testscaled.reshape(X_test.shape[0], X_test.shape[1], 1)

# model trend by linear regression
from sklearn import linear_model
modelLR = linear_model.LinearRegression()
XLR = np.array(df_train[['Time']])
yLR = np.array(df_train[['Passengers']])
modelLR.fit(XLR, yLR)
slope = modelLR.coef_ # the number of passengers is increasing per month on avarage

#%%
# Construct the LSTM model

modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seqlength, 1), return_sequences=False),
    tf.keras.layers.Dense(predictiontime)
    ])
modelLSTM.compile(
    loss='mse',
    optimizer = tf.optimizers.Adam(learning_rate = 0.001), 
    metrics = ['mae'])

modelLSTM.fit(X_scaledLSTM, y_scaled, epochs=200, batch_size = seqlength)

#%% Passengers = PrevPassengers + PassengerDiff + trend

predictionDiff = scalero.inverse_transform(modelLSTM.predict(X_testscaledLSTM[predictiontime - 1].reshape(1,12,1)))
prediction = np.zeros(13)
prediction[0] = df_test['Passengers'].iloc[-1-predictiontime]
for i in range(1,13):
    for j in range(1,13):
        prediction[j] = prediction[j-1] + predictionDiff[0][j-1] + slope
prediction = np.array(prediction[1:])

# Report a graph which shows your prediction along with the actual number of passengers. 
# Also report the mean absolute error of your prediction.

# Create another dataframe for plotting
df_pred = df_test[-12:]
df_pred['PassengerPred'] = prediction

# Create the plot
plt.plot(df['Month'].values, df['Passengers'].values, color='black', label='Actual number of passengers')
plt.plot(df_pred['Month'].values, df_pred['PassengerPred'], color='red', label = 'Prediction')
plt.grid()
plt.legend()
plt.show()

# Calculate the mean absolute error
from sklearn.metrics import mean_absolute_error
print('Mean absolute error in the test data is %.2f'
      % mean_absolute_error(df_pred['Passengers'].values, df_pred['PassengerPred'].values))