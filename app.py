import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("xgb_model.pkl")
columns = joblib.load("feature_columns.pkl")

st.title("⚡ Power Outage Predictor")
st.write("Enter values below to predict whether a power outage will occur.")

# Collect input from user
user_input = {}
for col in columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("🔴 Prediction: POWER OUTAGE" if prediction == 1 else "🟢 Prediction: POWER ON")