# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:44:46 2022

@author: schum
"""

#from requests import request
import pickle
import numpy as np
from flask import Flask, render_template, request
from utils import get_base_url



port = 12345
base_url = get_base_url(port)




# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')
    
    

@app.route(f'{base_url}')

def home():
    return render_template('index.html')
 

@app.route(f'{base_url}',methods=['POST'])

def predict():
    
    
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']
    
    
    features= [Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    
    loaded_model = pickle.load(open("rf_classifier.sav", "rb"))
    
    pred = loaded_model.predict((np.array([features])))
    
    if pred[0]==1:
        val="Diabetic"
    else:
        val="Not Diabetic"
    source = "The model classifies you as "+val+" "
    return render_template("index.html",pred=source)
   



if __name__ =="__main__":
    
    website_url = 'https://cocalc4.ai-camp.dev/'
    print(base_url)
    print(f'Try to open\n\n    https://cocalc4.ai-camp.dev/' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)