# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:19:56 2020

@author: Niteesh
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

pickle_in = open('cvd-prediction-svc-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome! to the Cardiovascular Diesase Predictor"

@app.route('/predict', methods=["GET"])
def predict_cvd():

    age = request.args.get('age')
    height = request.args.get('height')
    weight = request.args.get('weight')
    systolic_bp = request.args.get('systolic_bp')
    diastolic_bp = request.args.get('diastolic_bp')
    cholesterol = request.args.get('cholesterol')
    glucose = request.args.get('glucose')
    smoke = request.args.get('smoke')
    alcoholic = request.args.get('alcoholic')
    active = request.args.get('active')
    gender = request.args.get('gender')

    x = [[age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoke, alcoholic, active]]
    

    prediction = classifier.predict(x)
    if prediction:
        return 'Oops! You must have CVD, consult a doctor.'
    else:
        return "Yay! negative, no CVD."

@app.route('/predict_file', methods=["POST"])
def predict_cvd_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return str(list(prediction))




if __name__ == '__main__':
    app.run()
