import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table
from my_functions import testeo_general, clean_tester
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import mlflow
from mlflow import MlflowClient
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Definir los argumentos del DAG
default_args = {
    'owner': 'Group 1',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['xxxxx@gmail.com'],
    'retries': 1,
    'start_date': datetime(2024, 5, 20),
}

# Función para realizar el testeo
def test_data():

    engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')

    query = "SELECT * FROM raw_data"
    df1 = pd.read_sql(query, con=engine)

    query = "SELECT * FROM test_data"
    df2 = pd.read_sql(query, con=engine)

    p, t = testeo_general(df1, df2)
    print(p, t)

    return print("Test realizado") 


def drop_table(table_name):
    # Conexión a MySQL (en docker)
    engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')
    # engine = create_engine('mysql+pymysql://root:airflow@127.0.0.1:3306/db')
    metadata = MetaData()
    mi_tabla = Table(table_name, metadata)
    mi_tabla.drop(engine)


def t1():

    le = LabelEncoder()
    isolation_forest = IsolationForest(random_state=42)

    # conectar con mlflow y minio
    mlflow.set_tracking_uri("http://Mlflow:5000")

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://Minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'


    engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')
    query = "SELECT * FROM test_data"
    df_test = pd.read_sql(query, con=engine)

    # REVISAR LA FUNCION DE LIMPIEZA ################################################
    df_test = clean_tester(df_test)




    # Guardar los datos en MySQL
    df_test.to_sql('clean_data', con=engine, if_exists='append', index=False)
    drop_table('test_data')

    # Parte 2 llamar tod el conjunto y reentrene
    query = "SELECT * FROM clean_data"
    df = pd.read_sql(query, con=engine)

    X = df.drop(columns='price')
    y = df['price']
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42) 

    # Parametros del modelo
    parametros = {
        'xgb__n_estimators': randint(50, 200),
        'xgb__max_depth': randint(3, 10),
        'xgb__learning_rate': uniform(0.01, 0.3),
        'xgb__subsample': uniform(0.7, 0.3),
        'xgb__colsample_bytree': uniform(0.7, 0.3)
    }

    # generar modelo
    model = xgb.XGBRegressor()
    # Cercha de busqueda
    random_search = RandomizedSearchCV(model, 
                                    param_distributions=parametros, 
                                    n_iter=10, cv=3, n_jobs=-1, random_state=42)

    # Modelo
    random_search.fit(X_train, y_train)
    # Tomar el mejor modelo a registrar
    best_model = random_search.best_estimator_
    # Uso del conjunto de prueba para evaluar el modelo final
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)


    EXPERIMENT_NAME = "Classifier-Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)
    #agregaron para eliminar el resto
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, 
                           registered_model_name='modelo')  
    
    current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))

    print('inicia el experimento')

    model_name = 'XGB'
    RUN_NAME = f'Regression Experiment {model_name}'
    with mlflow.start_run(run_name=RUN_NAME):

        mlflow.log_metric(f"{model_name}_r2", r2)
        mlflow.log_metric(f"{model_name}_mae", mae)
        mlflow.log_metric(f"{model_name}_mse", mse)
        mlflow.log_metric(f"{model_name}_rmse", rmse)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"{model_name} model for regression")
        
        #log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=f"house_{model_name}_model",
            input_example=X_train,
            registered_model_name=f"tracking-house-{model_name}"
        )

        print('finaliza el experimento')

        mlflow.end_run() 

    client = MlflowClient()
    client.set_registered_model_tag("tracking-house-XGB", "task", "regression")

    print("Trained successfully.")


def t2():
    print('Final final no va más')


# Definir el DAG
dag = DAG(
    'test_data_dag',
    default_args=default_args,
    description='Comparar',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
test_data = PythonOperator(task_id='test_data', python_callable=test_data, dag=dag)
reentrenar = PythonOperator(task_id='t1', python_callable=t1, dag=dag)
noentrenar = PythonOperator(task_id='t2', python_callable=t1, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> test_data >> [t1, t2] >> end_operator
