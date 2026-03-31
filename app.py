import streamlit as st
import joblib
import numpy as np

# Load model and features
model = joblib.load("Diabetes_prediction.pkl")
features = joblib.load("features.pkl")

st.title("Diabetes Prediction App")

st.write("Enter patient details below:")

# Create input fields dynamically
user_input = []

for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ High risk of Diabetes")
    else:
        st.success("✅ Low risk of Diabetes")
