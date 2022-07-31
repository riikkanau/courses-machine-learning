# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:40:17 2021

@author: riikka.naumanen
"""

# no defined output variables, does not require annotated dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('Mall_customers.csv')

X = np.array(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

model = KMeans(n_clusters = 4)
model.fit(Xscaled)

df['Group'] = model.labels_

# colours for customer groups
colors = {0:'red', 1:'blue', 2:'green', 3:'magenta'}

# 3D coordinate system
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for i in range(0,4):
    # input variables in each axis
    x = df.loc[df['Group'] == i]['Age'].values
    y = df.loc[df['Group'] == i]['Annual Income (k$)'].values
    z = df.loc[df['Group'] == i]['Spending Score (1-100)'].values
    # plot each customer as a point
    ax.scatter(x, y, z, marker = 'o', s = 40, color = colors[i], label = 'Customer group '+str(i+1))

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
ax.legend()
plt.show()
