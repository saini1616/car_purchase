import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model and the scaler
model = joblib.load('car_model')

# Load the pre-fitted scaler
sc = joblib.load('scaler_model')  # Replace 'scaler_model' with the actual filename where the fitted scaler was saved.
sc1 = joblib.load('scaler_output')  # Assuming there's another scaler for output, load it similarly.

# Streamlit UI
st.title("Car Purchase Amount Predictions Using Machine Learning")

st.write("Enter the following details to predict car purchase amount:")

# User input fields (Gender, Age, Annual Salary, Credit Card Debt, Net Worth)
p1 = st.number_input("Gender (0 for Female, 1 for Male)", min_value=0, max_value=1, step=1)
p2 = st.number_input("Age", min_value=0)
p3 = st.number_input("Annual Salary", min_value=0.0)
p4 = st.number_input("Credit Card Debt", min_value=0.0)
p5 = st.number_input("Net Worth", min_value=0.0)

# Prediction button
if st.button('Predict'):
    # Prepare input for the model
    user_input = np.array([[p1, p2, p3, p4, p5]])

    # Scale input based on pre-fitted scaler
    scaled_input = sc.transform(user_input)

    # Make prediction
    result = model.predict(scaled_input)

    # Inverse transform the result to get original scale
    final_result = sc1.inverse_transform(result)

    # Display the result
    st.write(f"Car Purchase Amount: {final_result[0][0]}")

