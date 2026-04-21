import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# -------- LOAD MODEL & SCALER (same folder) --------
model = keras.models.load_model("netflix_churn_model.h5")
scaler = joblib.load("netflix_scaler.pkl")

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")

st.title("🎬 Netflix Customer Churn Prediction")
st.markdown("### Predict whether a customer is likely to churn")

st.divider()

# -------- INPUT SECTION --------
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 100, 25)
    watch_hours = st.number_input("Watch Hours", 0.0, 500.0, 50.0)
    monthly_fee = st.number_input("Monthly Fee", 0.0, 50.0, 10.0)

with col2:
    last_login_days = st.number_input("Last Login Days", 0, 100, 5)
    avg_watch_time = st.number_input("Avg Watch Time Per Day", 0.0, 10.0, 2.0)
    gender = st.selectbox("Gender", ["Male", "Female"])

# -------- ENCODING --------
gender = 0 if gender == "Male" else 1

# -------- DEFAULT VALUES (match training encoding) --------
region = 1
device = 2
subscription = 1
payment = 0
extra1 = 0
extra2 = 0

st.divider()

# -------- PREDICTION --------
if st.button("🔍 Predict Churn"):

    # Create full input (12 features)
    input_data = np.array([[
        age,
        watch_hours,
        last_login_days,
        monthly_fee,
        avg_watch_time,
        gender,
        region,
        device,
        subscription,
        payment,
        extra1,
        extra2
    ]])

    # Scale numeric features (first 5 only)
    input_data[:, :5] = scaler.transform(input_data[:, :5])

    # Prediction
    prediction = model.predict(input_data)[0][0]
    probability = prediction * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    # -------- METRICS --------
    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", f"{probability:.2f}%")

    if probability > 70:
        risk = "🔴 High Risk"
    elif probability > 40:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    col2.metric("Risk Level", risk)

    # Replace with your actual accuracy if needed
    col3.metric("Model Accuracy", "88%")

    st.divider()

    # -------- FINAL MESSAGE --------
    if prediction > 0.5:
        st.error("⚠️ This customer is likely to churn. Take action!")
    else:
        st.success("✅ This customer is likely to stay.")