#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:47:14 2022

@author: ladiyusuph
"""

import numpy as np
import joblib

#Loading the Model
loaded_model = joblib.load(open('trained_model.sav','rb'))

#Testing the loaded model with known data, to see if it can correctly label the data
input_data = (0.14710,1.0950,153.40,0.04904,25.38,17.33,184.60,2019.0,0.7119,0.2654)

#Converting the input dat to a numpy array
input_array = np.asarray(input_data)

#Reshaping the input array
input_reshaped = input_array.reshape(1,-1)

#Making the prediction
prediction = loaded_model.predict(input_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Cancer is Malignant')
else:
  print('The Cancer is Benign')