import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('anaconda/diabetes_model.pkl')
scaler = joblib.load('anaconda/scaler.pkl')


# Set page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    input, .stNumberInput input {
        background-color: #333;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Prediction App")
st.markdown("#### Enter the patient data below to check the likelihood of diabetes.")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 122)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 846)
bmi = st.number_input("BMI", 0.0, 67.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 0, 120, step=1)

if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    # If you used a scaler during training
    data = scaler.transform(data)
    result = model.predict(data)

    st.subheader("🧬 Prediction Result:")
    if result[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")
