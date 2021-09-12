# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 00:04:09 2021

@author: kolhe
"""


import os
import numpy as np
from flask import Flask, request, jsonify
import joblib
import pickle

os.chdir('/Users/Fathima Wasim')
print(os.getcwd())

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
model = joblib.load('linreg.pkl')
# takes 3 params

@app.route('/')
def home():
    """
    Uses GET Method
    """
    result_dic = {} # blank dict
    input_data = request.args.get('input')
    ls_data = eval(input_data)

    prediction = model.predict([np.array(ls_data)])
    output = prediction[0]
    result_dic["Input"] = ls_data
    result_dic["Predicted claim"] = output
    return jsonify(result_dic)


#@app.route('/hello')
#def hello():
#    return("hello")

# =============================================================================

@app.route('/predict_api',methods=['POST'])
def predict_api():
    """
    Uses POST Method
    """
    ls_data = request.get_json(force=True)
    print(ls_data)
    prediction = model.predict([np.array(ls_data)])

    output = prediction[0]
    return jsonify(output)
# =============================================================================

if __name__ == "__main__":
    app.run(debug=False)