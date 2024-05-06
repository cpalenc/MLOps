import streamlit as st
import requests

API_URL = 'http://fastapi:8002'

st.title('Interfaz con FastAPI')

# Definir un campo de entrada en Streamlit
name = st.text_input('Ingrese su nombre')

# Botón para enviar la solicitud a la API
if st.button('Saludar'):
    # Enviar la solicitud a la API
    response = requests.get(f'{API_URL}/hello', params={'name': name})
    # Mostrar la respuesta de la API en Streamlit
    if response.status_code == 200:
        st.success(response.json())
    else:
        st.error('Error al conectar con la API')
