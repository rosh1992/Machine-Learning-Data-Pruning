#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:19:18 2018

@author: rameshc
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'energydata.csv', delimiter=',')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:30])
X[:, 1:30] = imputer.transform(X[:, 1:28])

#Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train = mm_X.fit_transform(X_train)
X_test = mm_X.transform(X_test)
