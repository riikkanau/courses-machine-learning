# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:11:22 2021

@author: riikka.naumanen
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.DataFrame({'x':[1.00, 2.00, 3.00, 4.00, 5.00], 'y':[1.00, 2.00, 1.30, 3.75, 2.25]})

X = np.array(df[['x']])
y = np.array(df[['y']])

model = linear_model.LinearRegression()
model.fit(X, y)
prediction = model.predict(np.array([[6]]))

print(prediction)

plt.figure()
plt.scatter(X, y, label = 'Given data')
plt.scatter(6, [[prediction]], label = 'My prediction')
plt.title('Exercise 4')
plt.xlabel('X-axes')
plt.ylabel('Y-axes')
plt.legend(loc="upper left")
