import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
# from airflow.operators.dummy_operator import DummyOperator
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from airflow import DAG
from airflow.operators.python_operator import PythonOperator



def load_and_slip():
    # Conexión a la base de datos MySQL
    from sqlalchemy import create_engine, inspect
    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://user2:password2@mysql2/database2')

    query = "SELECT * FROM clean_data"
    # Leer los datos desde MySQL
    df = pd.read_sql(query, con=mysql_engine)
    X = df.drop(columns='price')
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42) 
    
    print("Datos limpios cargados desde MySQL")  

    return X_train, X_test, y_train, y_test


def best_model():

    X_train, X_test, y_train, y_test = load_and_slip()

    param_dist = {
        'xgb__n_estimators': randint(50, 200),
        'xgb__max_depth': randint(3, 10),
        'xgb__learning_rate': uniform(0.01, 0.3),
        'xgb__subsample': uniform(0.7, 0.3),
        'xgb__colsample_bytree': uniform(0.7, 0.3)
    }

    model = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(model, 
                                    param_distributions=param_dist, 
                                    n_iter=10, cv=3, n_jobs=-1, random_state=42)

    # Ajustar el modelo
    random_search.fit(X_train, y_train)

    # Mejor modelo después de la búsqueda
    best_model = random_search.best_estimator_
    best_model

    # Uso del conjunto de prueba para evaluar el modelo final
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return(best_model, mae, mse, rmse, r2)



def model_train():


    # conectar con mlflow y minio
    mlflow.set_tracking_uri("http://Mlflow:5000")

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'


    # X_train, X_test, y_train, y_test = load_and_slip()

    EXPERIMENT_NAME = "Classifier-Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)

    #agregaron para eliminar el resto
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, registered_model_name='modelo')    


    current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))

    print('inicia el experimento')

    model_name = 'XGB'
    RUN_NAME = f'Regression Experiment {model_name}'
    
    best_model_result,  mae, mse, rmse, r2 = best_model()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_metric(f"{model_name}_r2", r2)
        mlflow.log_metric(f"{model_name}_mae", mae)
        mlflow.log_metric(f"{model_name}_mse", mse)
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.sklearn.log_model(best_model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        model_details = mlflow.register_model(model_uri=model_uri, name=RUN_NAME)

    print("Trained successfully.")



with DAG (dag_id= 'nuevo_entrenamiento',
          description= "transform raw data and load to clean data",
          schedule_interval= "@once",
          start_date= datetime(2024,5,27)) as dag:
        
            t1 = PythonOperator (task_id="nuevoEntrenamiento",
                                python_callable=model_train)