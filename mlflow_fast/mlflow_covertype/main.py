from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from Penguins_val import Penguins

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc

app = FastAPI()

@app.post("/predict/")
def predict(data:Penguins):
    data = data.dict()
    elevation = data['elevation']
    horizontal_distance_to_roadways = data['horizontal_distance_to_roadways']
    hillshade_9am = data['hillshade_9am']
    horizontal_distance_to_fire_points = data['horizontal_distance_to_fire_points']

    columns = ['elevation', 'horizontal_distance_to_roadways', 'hillshade_9am', 'horizontal_distance_to_fire_points']

    # Estandarizar variables
    user_input = [elevation, horizontal_distance_to_roadways, hillshade_9am, horizontal_distance_to_fire_points]
    # user_input_scaled = func_transform(user_input)

    print('ok_load data')

    MLFLOW_TRACKING_URI = "http://Mlflow:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "tracking-cover-Decision Tree"
    model_version = 1

    print('ok_load')

    lr = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # Realizar la predicción
    # df_pred = pd.DataFrame(user_input)
    df_pred = pd.DataFrame([user_input], columns=columns)
    out_model = lr.predict(df_pred)

    print('ok_predict')
    sout = out_model[0]
    print(sout)
    print('##########')

    predicted = "Prediccion de especie: {}".format(sout)

    return {
        predicted
    }
