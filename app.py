import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model and scaler
model = joblib.load('car_model')
  # Assuming you have already trained your scaler
sc = joblib.load('scaler.pkl')  # Load the saved scaler

# Streamlit app title
st.title("Car Purchase Amount Predictions Using Machine Learning")

# Input fields
p1 = st.number_input("Gender (1 for Male, 0 for Female)", min_value=0.0, max_value=1.0, step=0.1)
p2 = st.number_input("Age", min_value=0.0)
p3 = st.number_input("Annual Salary", min_value=0.0)
p4 = st.number_input("Credit Card Debt", min_value=0.0)
p5 = st.number_input("Net Worth", min_value=0.0)

# Predict button
if st.button('Predict'):
    # Predict the car purchase amount
    input_data = np.array([[p1, p2, p3, p4, p5]])
    scaled_input = sc.transform(input_data)
    result = model.predict(scaled_input)

    # Display the result
    # Display the result
    st.write(f"Predicted Car Purchase Amount: ${result[0].item():,.1f}")  # Use .item() to get a scalar

