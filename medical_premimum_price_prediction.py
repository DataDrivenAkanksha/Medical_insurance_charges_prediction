#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load model and scaler
model =  joblib.load('Medical_insurance_prediction.pkl')
scaler = joblib.load(open("scaler1.pkl", "rb"))

# Set title
st.title("🩺 Medical Insurance Charge Predictor")

# Sidebar inputs
st.sidebar.header("Enter Patient Information")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
#bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.01, format="%.2f")
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Create raw input data
input_dict = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}

input_df = pd.DataFrame([input_dict])

# Apply same get_dummies as training
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Create empty row to match training columns
# (You must define the exact column order used in training)
expected_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                 'region_northwest', 'region_southeast', 'region_southwest']

for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # Add missing dummy columns

# Reorder columns to match model input
input_encoded = input_encoded[expected_cols]

# Scale the input
input_scaled = scaler.transform(input_encoded)

# Predict
if st.sidebar.button("Predict Charges"):
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Estimated Insurance Charges: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")





# In[ ]:




