import streamlit as st
from src.logger import logging
from src.utils import get_custom_dataframe
from src.pipeline.predict_pipeline import PredictPipeline

st.title("Maternal Health Risk Predictor")
st.markdown("### Please fill in the details below:")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    systolic_bp = st.number_input("Systolic BP", min_value=0, max_value=200, value=120)
    diastolic_bp = st.number_input("Diastolic BP", min_value=0, max_value=200, value=80)
    blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=0.0, max_value=30.0, value=5.6, format="%.1f")
    body_temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, format="%.1f")
    heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=72)

    submit = st.form_submit_button("Predict Maternal Risk Level")

if submit:
    input = get_custom_dataframe(age, systolic_bp,
                                 diastolic_bp, blood_sugar,
                                 body_temp, heart_rate)
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input)
    st.success(f"Predicted Maternal Risk Level: {prediction[0]}")