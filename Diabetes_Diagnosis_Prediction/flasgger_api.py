# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:45:56 2020

@author: Niteesh
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('diabetes-prediction-rfc-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome! to the Diabeties Predictor"

@app.route('/predict', methods=["GET"])
def predict_diabeties():
    
    """Diabeties Prediction
    ---
    parameters:
      - name: pregnancies
        in: query
        type: number
        required: true
      - name: glucose
        in: query
        type: number
        required: true
      - name: Blood Pressure
        in: query
        type: number
        required: true
      - name: Skin Thickness
        in: query
        type: number
        required: true
      - name: insulin
        in: query
        type: number
        required: true
      - name: Body Mass Index
        in: query
        type: number
        required: true
      - name: Diabetes Pedigree Function
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
            
    """
    pregnancies = request.args.get('pregnancies')
    glucose = request.args.get('glucose')
    bp = request.args.get('Blood Pressure')
    st = request.args.get('Skin Thickness')
    insulin = request.args.get('insulin')
    bmi = request.args.get('Body Mass Index')
    dpf = request.args.get('Diabetes Pedigree Function')
    age = request.args.get('age')

    x = [[pregnancies, glucose, bp, st, insulin, bmi, dpf, age]]
    x = StandardScaler().fit_transform(x)
    
    prediction = classifier.predict(x)
    if prediction:
        return 'Oops! You have diabetes.'
    else:
        return "Great! You don't have diabetes."

@app.route('/predict_file', methods=["POST"])
def predict_diabeties_file():
    
    """Diabeties Prediction
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values 
          
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return str(list(prediction))
    
if __name__ == '__main__':
    app.run()