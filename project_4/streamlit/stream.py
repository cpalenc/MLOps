import streamlit as st
import requests
import pandas as pd

# URL FastAPI
API_URL = "http://Fastapi:8000/predict"

# Title
st.title("Prediction of price")

# Num cols
col1, col2 = st.columns(2)

with col1:
     bed = st.number_input("bed", value=1)
     bath = st.number_input("bath", value=1)
     states = st.number_input("states", value=1)

with col2:
     acre_lot = st.number_input("acre_lot", step=0.1)
     house_size = st.number_input("house_size", step=0.1)

# Botón
if st.button("Predict"):
    # dictionary to send predict api
    input_data = {
        "bed": bed,
        "bath": bath,
        "acre_lot":acre_lot,
        "states":states,
        "house_size": house_size,
    } 

    # testeo
    st.write(input_data)
    response = requests.post('http://Fastapi:8000/predict/', json=input_data)
    st.write(response.json())
    prediction = response.json()
    st.write(f"Predicted Species: {prediction}")
