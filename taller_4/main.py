from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow

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