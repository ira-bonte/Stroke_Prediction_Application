# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 20:04:47 2025

@author: Abiman Barah JacQues
"""

# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved model
model = joblib.load('stroke_prediction_model.pkl')

# Initialize Flask app
app = Flask("Stroke Predicition")

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML form in templates folder

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create feature array
        features = np.array([[gender, age, hypertension, heart_disease,
                              ever_married, work_type, residence_type,
                              avg_glucose_level, bmi, smoking_status]])

        # Make prediction
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            result = 'High Risk of Stroke'
        else:
            result = 'Low Risk of Stroke'

        return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
