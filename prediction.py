# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:59:59 2021

@author: kzani
"""
from joblib import load
import numpy as np
import json

my_model = load('my_model.pkl')

def my_prediction(id):
    my_model = load('my_model.pkl')
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    prediction = my_model.predict(dummyT)
    if prediction == 0: status_str = " not "
    else: status_str = " "
    str = f"The predicted value is {prediction[0]}, therefore this booking will likely{status_str}be cancelled."
    return str
    