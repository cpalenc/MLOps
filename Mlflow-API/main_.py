from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
import mlflow
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from json import dump, load
with open('/data_model/covertype.json') as f:
    data = load(f)
df = pd.DataFrame(data['data'])
# Assign column names because json just contains data without headers (column names)
column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type',
                'Cover_Type']
df.columns = column_names

# Set the target values
y = df['Cover_Type']#.values

# Set the input values
df.drop('Cover_Type', axis=1, inplace=True)
X = df#.values
X_train, X_test, y_train, y_test = train_test_split(X, y)
column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                        ["Wilderness_Area", "Soil_Type"]),
                                      remainder='passthrough') # pass all the numeric values through the pipeline without any changes.
pipe = Pipeline(steps=[("column_trans", column_trans),("scaler", StandardScaler(with_mean=False)), ("RandomForestClassifier", RandomForestClassifier())])
param_grid =  {'RandomForestClassifier__max_depth': [1,2,3,10], 'RandomForestClassifier__n_estimators': [10,11]}

search = GridSearchCV(pipe, param_grid, n_jobs=2)
import os
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

# connect to mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_tracking_examples")

mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name="modelo1")

with mlflow.start_run(run_name="autolog_pipe_model_reg") as run:
    search.fit(X_train, y_train)


# Cargar el modelo desde MLflow
model_production_uri = "models:/{model_name}/production".format(model_name="modelo1")
loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)

# Definir la estructura de entrada de la API con valores por defecto
class InputData(BaseModel):
    Elevation: float = 0.0
    Aspect: float = 0.0
    Slope: float = 0.0
    Horizontal_Distance_To_Hydrology: float = 0.0
    Vertical_Distance_To_Hydrology: float = 0.0
    Horizontal_Distance_To_Roadways: float = 0.0
    Hillshade_9am: float = 0.0
    Hillshade_Noon: float = 0.0
    Hillshade_3pm: float = 0.0
    Horizontal_Distance_To_Fire_Points: float = 0.0
    Wilderness_Area: int = 0
    Soil_Type: int = 0

# Crear la aplicación FastAPI
app = FastAPI()

# Endpoint para la predicción con valores por defecto
@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convertir los datos de entrada a un DataFrame para la predicción
        input_data = [[data.Elevation, data.Aspect, data.Slope, data.Horizontal_Distance_To_Hydrology,
                       data.Vertical_Distance_To_Hydrology, data.Horizontal_Distance_To_Roadways,
                       data.Hillshade_9am, data.Hillshade_Noon, data.Hillshade_3pm,
                       data.Horizontal_Distance_To_Fire_Points, data.Wilderness_Area, data.Soil_Type]]
        prediction = loaded_model.predict(input_data)[0]  # Realizar la predicción
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
