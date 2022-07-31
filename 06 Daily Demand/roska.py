# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:03:26 2021

@author: riikka.naumanen
"""

# tänne poistettu koodi, jota en halunnut vielä heittää roskiin

# Choose days 0 – 250 as your training data 
Xtrain = np.array(df.iloc[:250,:1])     # Day
ytrain = np.array(df.iloc[:250, 1:])    # Demand

# days 251 – 300 as your test data
Xtest = np.array(df.iloc[250:,:1])     # Day
ytest = np.array(df.iloc[250:, 1:])    # Demand