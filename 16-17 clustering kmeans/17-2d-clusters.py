# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:32:37 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sns

df = pd.read_csv('2dclusters.csv', sep = ';', header= None)

X = np.array(df)

scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

model = KMeans(n_clusters = 10)
model.fit(Xscaled)

df.columns = ['x', 'y']
df['Group'] = model.labels_

# https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python
facet = sns.lmplot(data=df, x='x', y='y', hue='Group', fit_reg=False, legend=True) #, legend_out=True)
