from airflow import DAG 
from airflow.utils.dates import days_ago
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Definir los argumentos del DAG
default_args = {
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    '3.Entrenamiento_de_modelo',
    default_args=default_args,
    description='A Machine Learning workflow for price prediction',
    schedule_interval=None,
    start_date=days_ago(1),
    #tags=['ml', 'diabetes'],
)



# Función para crear el modelo
def make_model():
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'


    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://user2:password2@mysql2/database2')
    query = "SELECT * FROM clean_data;"
    df = pd.read_sql(query, con=mysql_engine)
    
    X = df.drop(columns='price')
    y = df['price']
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42) 

    
    model_name = 'tracking-house-XGB'
    mlflow.set_tracking_uri("http://Mlflow:5000")
    mlflow.set_experiment('PricePredictionExperiment')
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, registered_model_name=model_name)    
    
    current_experiment=dict(mlflow.get_experiment_by_name('PricePredictionExperiment'))    
    experiment_id=current_experiment['experiment_id']

    print('inicia el experimento')

    
    RUN_NAME = f'Regression Experiment {model_name}'
    params = {'n_estimators':18}
    
    
    #with mlflow.start_run(run_name=RUN_NAME) as run:


    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    signature = infer_signature(X_test, y_pred)
             
        
        # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    
    
        
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
      
    model_path = 'model'
    model.save_model(model_path)
    mlflow.log_artifact(model_path)


    print('finaliza el experimento')



    return print("Trained successfully.")



# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
make_modelo = PythonOperator(task_id='make_model', python_callable=make_model, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> make_modelo >> end_operator

