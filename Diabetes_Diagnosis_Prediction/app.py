# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:45:56 2020

@author: Niteesh
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
# from imblearn.ensemble import BalancedBaggingClassifier


app = Flask(__name__)

pickle_in = open('diabetes_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    for rendering result on HTML GUI
    """

    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    my_prediction = classifier.predict(final_features)

    if my_prediction:
        return render_template('index.html', prediction=my_prediction)
    else:
        return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
