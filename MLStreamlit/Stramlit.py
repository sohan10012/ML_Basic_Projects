#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:50:51 2025
@author: Sohan
"""

import numpy as np
import pickle
import streamlit as st

# Load trained model
loaded_model = pickle.load(open('/home/Sohan/Downloads/MLStreamlit/trained_model.sav', 'rb'))

# Prediction function
def dp(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return '✅ The person is **not diabetic**'
    else:
        return '⚠️ The person is **diabetic**'

# Main app
def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f7f9fc;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
        }
        .stNumberInput label {
            font-weight: bold;
            color: #34495e;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #27ae60;
            transform: scale(1.05);
        }
        .stSuccess {
            font-size: 20px;
            text-align: center;
            color: #2c3e50 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 Diabetes Prediction Form")

    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, format="%.1f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

    # Prediction
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

    if st.button("🔍 Get Diabetes Test Result"):
        diagnosis = dp(input_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
