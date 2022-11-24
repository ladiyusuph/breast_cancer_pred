#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:39:32 2022

@author: ladiyusuph
"""

import numpy as np
import joblib
import streamlit as st

#Loading the Model
loaded_model = joblib.load(open('trained_model.sav','rb'))

def breast_cancer_pred(input_data):
   
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
      
      
def main():
    
    
    #Giving the app a title
    st.title('Breast Cancer Prediction App')
    #Getting input from user
    concave = st.text_input('Mean concave points')
    radius_err = st.text_input('Radius Error')
    area_err = st.text_input('Area error')
    compactness_err = st.text_input('Compactness Error')
    worst_radius = st.text_input('Worst Radius')
    worst_texture = st.text_input('Worst Texture')
    worst_perimeter = st.text_input('Worst Perimeter')
    worst_area = st.text_input('Worst Area')
    worst_concavity = st.text_input('Worst Concavity')
    worst_concave = st.text_input('Worst Concave Points')
    
    #Code for prediction
    diagnosis = ""
    
    if st.button('Test Result'):
        diagnosis = breast_cancer_pred([concave,radius_err,area_err,compactness_err,worst_radius,worst_texture,\
                                        worst_perimeter,worst_area,worst_concavity,worst_concave])
        
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()