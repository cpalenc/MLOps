import streamlit as st
import requests
import pandas as pd

# URL FastAPI
API_URL = "http://Fastapi:8000/predict"

# Title
st.title("Prediction of Re-admitted")

# Num cols
col1, col2 = st.columns(2)

with col1:
     age = st.number_input("Age", value=5)
     discharge_disposition_id = st.number_input("Discharge disposition id", value=25)
     time_in_hospital = st.number_input("Time in hospital", value=1)
     num_lab_procedures = st.number_input("Num lab procedures", value=41)
     num_procedures = st.number_input("Num procedures", value=0)

with col2:
     number_inpatient = st.number_input("Number inpatient", value=0)
     diag_1 = st.number_input("diag 1", value=253)
     diag_2 = st.number_input("diag 2", value=255)
     diag_3 = st.number_input("diag 3", value=255)
     number_diagnoses = st.number_input("Number diagnoses", value=1)

# Botón
if st.button("Predict"):
    # dictionary to send predict api
    input_data = {
        "age": age,
        "discharge_disposition_id": discharge_disposition_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
    } 

    # testeo
    st.write(input_data)
    response = requests.post('http://Fastapi:8000/predict/', json=input_data)
    st.write(response.json())
    prediction = response.json()
    st.write(f"Predicted Re-admitted: {prediction}")
