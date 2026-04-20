import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# Load model & scaler
model = keras.models.load_model("netflix_churn_model.h5")
scaler = joblib.load("netflix_scaler.pkl")

st.title("Netflix Churn Prediction")

st.write("Enter user details:")

# Inputs (same features you trained on!)
age = st.number_input("Age", 10, 100, 25)
watch_hours = st.number_input("Watch Hours", 0.0, 500.0, 50.0)
last_login_days = st.number_input("Last Login Days", 0, 100, 5)
monthly_fee = st.number_input("Monthly Fee", 0.0, 50.0, 10.0)
avg_watch_time = st.number_input("Avg Watch Time Per Day", 0.0, 10.0, 2.0)

if st.button("Predict"):
    
    # Arrange input (VERY IMPORTANT: same order as training)
    input_data = np.array([[age, watch_hours, last_login_days, monthly_fee, avg_watch_time]])
    
    # Scale
    input_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data)[0][0]
    
    st.write(f"Churn Probability: {prediction:.2f}")
    
    if prediction > 0.5:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")