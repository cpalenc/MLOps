import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table
from my_functions import testeo_general,clean_tester
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import mlflow
from mlflow import MlflowClient
from scipy.stats import randint, uniform
from mlflow.models import infer_signature
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

def test_data():

    engine = create_engine('mysql://user1:password1@mysql1/database1')

    query = "SELECT * FROM raw_data_bienes_raices"
    df1 = pd.read_sql(query, con=engine)

    query = "SELECT * FROM test_raw_data_bienes_raices"
    df2 = pd.read_sql(query, con=engine)

    p = testeo_general(df1, df2)
    print(p)

    return p  

def drop_table(table_name): # elimina test_data
    engine = create_engine('mysql://user1:password1@mysql1/database1')
    metadata = MetaData()
    mi_tabla = Table(table_name, metadata)
    mi_tabla.drop(engine)



def t1():


    # conectar con mlflow y minio
    mlflow.set_tracking_uri("http://Mlflow:5000")

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://Minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    
############ esto se deberia hacer a partir del dag de transformacion y carga ################
    
    engine1 = create_engine('mysql://user1:password1@mysql1/database1')
    query = "SELECT * FROM test_raw_data_bienes_raices"
    df_test = pd.read_sql(query, con=engine1)
    df_carga = df_test.copy()
    #df_test = clean_tester(df_test)
    df_carga.to_sql('raw_data_bienes_raices', con=engine1, if_exists='append', index=False)
    
    query_raw = "SELECT * FROM raw_data_bienes_raices"
    df_raw_read = pd.read_sql(query_raw, con=engine1)
    
    df_raw_read = clean_tester(df_raw_read)
    
#############################################################################################
    
    engine_clean = create_engine('mysql://user2:password2@mysql2/database2')
    df_raw_read.to_sql('clean_data', con=engine_clean, if_exists='append', index=False)
    

   
####################### Acá debería llamar al DAG de entrenamiento ############################

    try:
        engine = create_engine('mysql://user2:password2@mysql2/database2')
        query = "SELECT * FROM clean_data"
        df = pd.read_sql(query, con=engine)
    except Exception as e:
        print(f'Error al cargar los datos: {e}')
        return 
        
    #drop_table('test_raw_data_bienes_raices')    
    
    num_samples = min(df.shape[0],50000)
    df = df.sample(n=num_samples, replace = True, random_state=42)
    print(df.shape)

    X = df.drop(columns='price')
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42) 

    # Parametros del modelo
    parametros = {
        'xgb__n_estimators':[50, 100,200],
        'xgb__max_depth': [3, 5,7, 10],
        'xgb__learning_rate': [0.01, 0.01, 0.3]#,
        #'xgb__subsample': uniform(0.7, 0.3),
        #'xgb__colsample_bytree': uniform(0.7, 0.3)
    }
    
    
    experiment_name = 'PricePrediction-Experiment'
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(experiment_name)
    
    
    
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('PricePrediction-Experiment')
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, registered_model_name='modelo_test')    
    
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))    
    experiment_id=current_experiment['experiment_id']

    print('inicia el experimento')
    
    
    
    


    model_name = 'XGB'
    RUN_NAME = f'Regression Experiment {model_name}'
    params = parametros
    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME):

        model = xgb.XGBRegressor()
        random_search = RandomizedSearchCV(model, 
                                    param_distributions=parametros, 
                                    n_iter=10, cv=3, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test) 

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

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
    '4.retrain_data_dag',
    default_args=default_args,
    description='Comparison between data batchs',
    schedule_interval=None
)


# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
test_data_task = PythonOperator(task_id='test_data', python_callable=test_data, dag=dag)
reentrenar = PythonOperator(task_id='t1', python_callable=t1, dag=dag)
noentrenar = PythonOperator(task_id='t2', python_callable=t2, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> test_data_task >> [reentrenar, noentrenar] >> end_operator
