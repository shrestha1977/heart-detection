import streamlit as st
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import json

# Page setup
st.set_page_config(page_title="Heart Risk Checker", page_icon="❤️", layout="centered")

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_heart = load_lottie("heartbeat.json")

# Custom styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif;
    }
    .dark-theme {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.toggle("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("<script>document.body.classList.add('dark-theme')</script>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Attack Risk Predictor</h1>", unsafe_allow_html=True)
st_lottie(lottie_heart, height=200, key="heart")
st.markdown("<h4 style='text-align: center;'>Predict your heart health with AI</h4>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
    trestbps = st.number_input("Resting BP", 80, 200)
    chol = st.number_input("Cholesterol", 100, 600)
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No (0)", "Yes (1)"])
    restecg = st.selectbox("ECG Results", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
    thalach = st.number_input("Max Heart Rate", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])

st.markdown("---")

if st.button("🧠 Predict Risk"):
    input_data = [
        age,
        int(sex[-2]),
        int(cp[-2]),
        trestbps,
        chol,
        int(fbs[-2]),
        int(restecg[-2]),
        thalach,
        int(exang[-2])
    ]
    scaled = scaler.transform([input_data])
    pred = model.predict(scaled)

    if pred[0] == 1:
        st.error("⚠️ High risk of heart attack. Please consult a doctor.")
    else:
        st.success("✅ Low risk. Keep living healthy!")

st.markdown("---")
st.markdown("<div style='text-align:center; font-size:13px;'>Made with ❤️ using Streamlit & ML</div>", unsafe_allow_html=True)
