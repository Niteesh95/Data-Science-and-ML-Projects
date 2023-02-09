# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:45:56 2020

@author: Niteesh
"""

from flask import Flask, request, jsonify, render_template, url_for
# import pandas as pd
# import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

pickle_in = open('./src/CVD_RF_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    for rendering result on HTML GUI
    """

    #age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    #systolic_bp = int(request.form['systolic_bp'])
    #diastolic_bp = int(request.form['diastolic_bp'])
    #cholesterol = int(request.form['cholesterol'])
    #glucose = int(request.form['glucose'])
    #smoke = int(request.form['smoke'])
    #alcoholic = int(request.form['alcoholic'])
    #active = int(request.form['active'])
    #gender = int(request.form['gender'])
    bmi = weight / (height**2)
    features = [[x for x in request.form.values()]]
    features[0].append(bmi)

    #final_features = [[age,gender,height,weight,systolic_bp,diastolic_bp,cholesterol,glucose,smoke,alcoholic,active]]
    my_prediction = classifier.predict(features)

    if my_prediction:
        return render_template('index.html', prediction_text="CVD Diagnosed!!!")
    else:
        return render_template('index.html', prediction_text="NO CVD Diagnosed!")


if __name__ == '__main__':
    app.run(debug=True)
