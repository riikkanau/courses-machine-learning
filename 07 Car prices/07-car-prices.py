# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:29:27 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


df = pd.read_csv('Car details v3.csv')
df.drop(['fuel', 'torque'], axis = 1, inplace = True)

df.dropna(inplace = True)
df.reset_index(drop = True, inplace = True)


# check what values 
# check = df['owner'].unique()

df['engine'] = df['engine'].str[:-3].astype(float)

# mileage -> Petrol & Diesel: kmpl, CNG & LPG: km/kg. 
df_mileage = df['mileage'].str.split(' ', expand = True)
df['mileage_amount'] = df_mileage[0].astype(float) 
df['mileage_unit'] = df_mileage[1]
df = df.drop(df[df.mileage_unit == 'km/kg'].index) # tiputetaan kaikki kaasuautot, koska niitä suhteessa vähän (87 kpl)


df = df.drop(df[df.max_power.str.len() <= 4].index) # drop lines where there is only unit, not amount
df.reset_index(drop = True, inplace = True) # reset index, jotta slice mahdollinen
df['max_power'] = df['max_power'].str[:-4].astype(float)

# drop some outliers
df = df.drop(df[df.km_driven >= 500000].index)
df = df.drop(df[df.selling_price >= 10000000].index)

#%%

X = np.array(df[['year', 'km_driven', 'mileage_amount', 'engine', 'max_power', 'seats']])

# input variable values are scaled to be able to report the relative importance of each input variable
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)


#%% OneHot encoding

# seller_type
Xseller_type = pd.get_dummies(df['seller_type'])
Xscaled = np.concatenate((Xscaled, Xseller_type), axis = 1)

# transmission
Xtransmission = pd.get_dummies(df['transmission'])
Xscaled = np.concatenate((Xscaled, Xtransmission), axis = 1)

# owner 
# mietin olisiko voinut olla järjestysasteikko ja olisi voinut laittaa numeroiksi yhteen sarakkeeseen, mutta testiajoautosta ei pysty sanomaan miten tulisi järjestää suhteessa omistajien määrään, joten OneHot tälle myös
Xowner = pd.get_dummies(df['owner'])
Xscaled = np.concatenate((Xscaled, Xowner), axis = 1)


#%% Output variable scaling

y = np.array(df[['selling_price']])

scalero = preprocessing.StandardScaler()
yscaled = scalero.fit_transform(y)

#%% Separate 30 % of the data randomly to a test data

Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, yscaled, test_size = 0.3)

#%%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation = tf.nn.relu, input_shape = (Xtrain.shape[1],)), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(30, activation = tf.nn.tanh),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(30, activation = tf.nn.relu),
    tf.keras.layers.Dense(ytrain.shape[1])
    
    ])

model.compile(loss = 'mse', 
              optimizer = tf.optimizers.Adam(learning_rate = 0.001), 
              metrics = ['mae']) 

model.fit(Xtrain, ytrain, validation_data = (Xtest, ytest), epochs = 50, batch_size = 10)

#%%

# Predicting and scaling the predictions back
trainprediction = scalero.inverse_transform(model.predict(Xtrain))
testprediction = scalero.inverse_transform(model.predict(Xtest))

# Scaling back the input variables that were scaled and concatenating with the ones that weren't scaled
Xtraincont = scaler.inverse_transform(Xtrain[:,0:6])
Xtrain = np.concatenate((Xtraincont, Xtrain[:,6:]), axis = 1)
Xtestcont = scaler.inverse_transform(Xtest[:,0:6])
Xtest = np.concatenate((Xtestcont, Xtest[:,6:]), axis = 1)

ytrain = scalero.inverse_transform(ytrain)
ytest = scalero.inverse_transform(ytest)

print('Mean absolute error in train data %.2f'
      % mean_absolute_error(ytrain, trainprediction))

print('Mean absolute error in test data %.2f'
      % mean_absolute_error(ytest, testprediction))

#%% Plots

# whole data 
plt.figure()
plt.scatter(Xtest[:,1], ytest, label = 'Actual selling price')
plt.scatter(Xtest[:,1], testprediction, label = 'Predicted selling price')
plt.title('Test data: car prices')
plt.xlabel('Driven km')
plt.ylabel('Selling price (Indian rupees)')
plt.legend()

# sample

df_test = pd.DataFrame()
df_test['km_driven'] = Xtest[:,1]
df_test['actual_price'] = ytest
df_test['predicted_price'] = testprediction

dfsample = df_test.sample(300)

plt.figure()
plt.scatter(dfsample['km_driven'], dfsample['actual_price'], label = 'Actual selling price')
plt.scatter(dfsample['km_driven'], dfsample['predicted_price'], label = 'Predicted selling price')
plt.title('Test data: sample of car prices')
plt.xlabel('Driven km')
plt.ylabel('Selling price (Indian rupees)')
plt.legend()

