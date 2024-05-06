import streamlit as st
import requests
import pandas as pd

# URL FastAPI
API_URL = "http://Fastapi:8000/predict"

# Title
st.title("Prediction of Land Cover Type")

# Num cols
col1, col2 = st.columns(2)

with col1:
     elevation = st.number_input("Elevation", value=2998)
     horizontal_distance_to_roadways = st.number_input("Horizontal Distance To Roadways", value=319)

with col2:
     hillshade_9am = st.number_input("Hillshade 9am", value=233)
     horizontal_distance_to_fire_points = st.number_input("Horizontal Distance To Fire Points", value=955)

# Botón
if st.button("Predict"):
    # dictionary to send predict api
    # input_dict = {
    #     "Elevation": elevation,
    #     "Horizontal_Distance_To_Roadways": horizontal_distance_to_roadways,
    #     "Hillshade_9am": hillshade_9am,
    #     "Horizontal_Distance_To_Fire_Points": horizontal_distance_to_fire_points,
    # } 
    input_data = {
        "elevation": elevation,
        "horizontal_distance_to_roadways": horizontal_distance_to_roadways,
        "hillshade_9am": hillshade_9am,
        "horizontal_distance_to_fire_points": horizontal_distance_to_fire_points,
    } 

    # testeo
    st.write(input_data)
    response = requests.post('http://Fastapi:8000/predict/', json=input_data)
    st.write(response.json())
    prediction = response.json()
    st.write(f"Predicted Species: {prediction}")

    # # Predict
    # response = requests.post(API_URL, json=input_dict)
    # print(response)
    # print("#########")
    
    # if response.status_code == 200:
    #     prediction = response.json()["Prediction"]

    #     print(prediction)
    #     print("#########")

    #     st.success(f"The prediction is: {prediction}")
    # else:
    #     st.error("Mistake 501")






