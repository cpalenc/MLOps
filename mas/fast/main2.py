import pandas as pd
from sqlalchemy import create_engine
import mlflow
from fastapi import FastAPI
from Penguins_val import Penguins 
# import mlflow.pyfunc

app = FastAPI()

engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')

@app.post("/predict/")
def predict(data: Penguins):
    bed = data.dict()
    bath = data['sepal_length']
    acre_lot = data['sepal_width']
    street = data['petal_length']
    house_size = data['petal_width']

    columns = ['bed', 'bath', 'acre_lot', 'street', 'house_size']

    # Estandarizar variables
    user_input = [bed, bath, acre_lot, street, house_size]
    # user_input_scaled = func_transform(user_input)

    print('ok_load data')

    MLFLOW_TRACKING_URI = "http://Mlflow:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "tracking-cover-knn"
    model_version = 1

    print('ok_load model')

    lr = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    df_pred = pd.DataFrame([user_input], columns=columns)
    out_model = lr.predict(df_pred)[0]
    predicted_species = "Prediccion de especie: {}".format(out_model)

    print('ok_prediccion')

    # Save user input and prediction to database
    df_salida = df_pred.copy()
    df_salida['pred'] = out_model

    # save inputs and predic in new table
    df_salida.to_sql('penguin_data', con=engine, if_exists='append', index=False)

    return {
        "predicted": predicted_species
    }
