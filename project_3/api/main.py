from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from Penguins_val import Penguins
from my_functions import func_transform

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc

app = FastAPI()

@app.post("/predict/")
def predict(data:Penguins):
    data = data.dict()
    age = data['age']
    discharge_disposition_id = data['discharge_disposition_id']
    time_in_hospital = data['time_in_hospital']
    num_lab_procedures = data['num_lab_procedures']
    num_procedures = data['num_procedures']
    number_inpatient = data['number_inpatient']
    diag_1 = data['diag_1']
    diag_2 = data['diag_2']
    diag_3 = data['diag_3']
    number_diagnoses = data['number_diagnoses']

    columns = ['age', 'discharge_disposition_id', 'time_in_hospital', 'num_lab_procedures', 
               'num_procedures', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses']

    # Estandarizar variables
    user_input = [age, discharge_disposition_id, time_in_hospital, num_lab_procedures,
                  num_procedures, number_inpatient, diag_1, diag_2, diag_3, number_diagnoses]
    user_input_scaled = func_transform(user_input)

    print('ok_load data')

    MLFLOW_TRACKING_URI = "http://Mlflow:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "tracking-readmitted-RF"
    model_version = 1

    print('ok_load')

    lr = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # Realizar la predicción
    df_pred = pd.DataFrame([user_input_scaled], columns=columns)
    out_model = lr.predict(df_pred)

    print('ok_predict')
    sout = out_model[0]
    print(sout)
    print('##########')

    predicted = "Predicted Re-admitted: {}".format(sout)

    return {
        predicted
    }
