import pandas as pd
from sqlalchemy import create_engine
import mlflow
from fastapi import FastAPI
import mlflow.pyfunc
from pydantic import BaseModel

app = FastAPI()

class input_chars(BaseModel):
    bed: float = 8.0
    bath: float = 2.0
    acre_lot: float = 0.09
    state: int = 10
    house_size: float = 1409.0

engine = create_engine('mysql://user1:password1@mysql1/database1')

@app.post("/predict/")
def predict(data: input_chars):
    data = data.dict()
    bed = data['bed']
    bath = data['bath']
    acre_lot = data['acre_lot']
    state = data['state']
    house_size = data['house_size']

    # Cargar el modelo para verificar los nombres de las características
    MLFLOW_TRACKING_URI = "http://Mlflow:5000"
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.search_model_versions("name='modelo_test'")
    latest_version = sorted(latest_versions, key=lambda x: x.creation_timestamp, reverse=True)[0]
    model_uri = f"models:/tracking-price-XGBoost/{latest_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Obtener los nombres de las características esperadas por el modelo
    expected_feature_names = model.metadata.get_input_schema().input_names()
    print(f"Expected feature names: {expected_feature_names}")

    # Ajustar los nombres de las características en el DataFrame
    user_input = [bed, bath, acre_lot, state, house_size]
    columns = ['bed', 'bath', 'acre_lot', 'state', 'house_size']
    df_pred = pd.DataFrame([user_input], columns=columns)

    # Cambiar el nombre de la columna si es necesario
    if 'state' in expected_feature_names and 'states' not in expected_feature_names:
        df_pred.rename(columns={'states': 'state'}, inplace=True)
    elif 'states' in expected_feature_names and 'state' not in expected_feature_names:
        df_pred.rename(columns={'state': 'states'}, inplace=True)

    print(df_pred.head())

    # Hacer la predicción
    prediction = model.predict(df_pred)

    print(prediction)

    predicted_species = "Prediccion de precio: {}".format(prediction)

    print('ok_prediccion')

    # Guardar entrada de usuario y predicción en la base de datos
    df_salida = df_pred.copy()
    df_salida['pred'] = prediction

    # Guardar entrada y predicción en nueva tabla
    df_salida.to_sql('user_data', con=engine, if_exists='append', index=False)

    return {
        "predicted": predicted_species
    }
