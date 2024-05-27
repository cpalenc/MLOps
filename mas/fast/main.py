from fastapi import FastAPI
import pandas as pd
from Penguins_val import Penguins

import mlflow
import mlflow.pyfunc

app = FastAPI()

@app.post("/predict/")
def predict(data:Penguins):
    
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    columns = ['sepal_length','sepal_width', 'petal_length', 'petal_width']

    # Estandarizar variables
    user_input = [sepal_length, sepal_width, petal_length, petal_width]
    # user_input_scaled = func_transform(user_input)

    print('ok_load data')

    MLFLOW_TRACKING_URI = "http://Mlflow:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "tracking-cover-knn"
    model_version = 1

    print('ok_load')

    lr = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # Realizar la predicción
    # df_pred = pd.DataFrame(user_input)
    df_pred = pd.DataFrame([user_input], columns=columns)
    out_model = lr.predict(df_pred)[0]
    print('ok_predict')
    print(out_model)
    print('##########')

    predicted = "Prediccion de especie: {}".format(out_model)

    return {
        predicted
    }
