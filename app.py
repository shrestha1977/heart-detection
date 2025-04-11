import streamlit as st
import pandas as pd
import pickle
from streamlit_lottie import st_lottie
import json

# Page config
st.set_page_config(page_title="Heart Attack Prediction", page_icon="❤️", layout="centered")

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load Lottie animation
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_heart = load_lottie("heartbeat.json")

# Custom fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("❤️ Heart Attack Risk Predictor")
st_lottie(lottie_heart, speed=1, height=200)

st.markdown("Enter your health information to predict heart attack risk.")

# Dark mode toggle
dark_mode = st.checkbox("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #1e1e1e;
            color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)

# Input form (13 features)
with st.form("heart_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100)
        sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.number_input("ST depression", 0.0, 6.0, step=0.1)
        slope = st.selectbox("Slope of the peak exercise ST", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# Predict
if submitted:
    input_data = [
        age,
        int(sex[-2]),
        int(cp[-2]),
        trestbps,
        chol,
        int(fbs[-2]),
        int(restecg[-2]),
        thalach,
        int(exang[-2]),
        oldpeak,
        int(slope[-2]),
        ca,
        int(thal[-2])
    ]

    scaled = scaler.transform([input_data])
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack!")
    else:
        st.success("✅ Low Risk — Keep up the good health!")

