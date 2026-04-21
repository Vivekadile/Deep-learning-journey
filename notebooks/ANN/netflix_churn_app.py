import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# -------- LOAD MODEL --------
model = keras.models.load_model("netflix_churn_model.h5", compile=False)
scaler = joblib.load("netflix_scaler.pkl")

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")

st.title("🎬 Netflix Customer Churn Prediction")
st.markdown("### Smart insights to prevent customer churn")

st.write("---")

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

# -------- DEFAULT VALUES --------
region = 1
device = 2
subscription = 1
payment = 0
extra1 = 0
extra2 = 0

st.write("---")

# -------- PREDICTION --------
if st.button("🔍 Predict Churn"):

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

    # Scale numeric features
    input_data[:, :5] = scaler.transform(input_data[:, :5])

    prediction = model.predict(input_data)[0][0]
    probability = float(prediction) * 100

    st.write("---")
    st.subheader("📊 Prediction Result")

    # -------- PROBABILITY --------
    st.metric("Churn Probability", f"{probability:.2f}%")
    st.progress(int(probability))

    # -------- RISK LEVEL --------
    if probability > 70:
        risk = "🔴 High Risk"
        message = "Immediate action required!"
    elif probability > 40:
        risk = "🟠 Medium Risk"
        message = "Monitor user behavior closely."
    else:
        risk = "🟢 Low Risk"
        message = "Customer is stable."

    st.write(f"### Risk Level: {risk}")
    st.info(message)

    st.write("---")

    # -------- FINAL RESULT --------
    if prediction > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    # -------- RECOMMENDATIONS --------
    st.subheader("💡 Suggested Actions")
    st.write("Raw prediction:", prediction)

    if prediction > 0.5:
        st.write("""
        - Offer discounts or special plans  
        - Send personalized recommendations  
        - Improve engagement (notifications/emails)  
        - Provide better content suggestions  
        """)
    else:
        st.write("""
        - Maintain engagement  
        - Recommend trending content  
        - Offer loyalty rewards  
        """)