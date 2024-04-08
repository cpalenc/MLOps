from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sklearn.linear_model import LinearRegression
import mlflow
import pandas as pd
from sqlalchemy import create_engine

def train_model_and_log_to_mlflow():
    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://airflow:airflow@mysql/airflow')
    df = pd.read_sql_table('dataset_table', con=mysql_engine)
    X = df.drop(columns=["Cover Type"])
    y = df["Cover Type"]
    model = LinearRegression()
    model.fit(X, y)
    
    mlflow.set_tracking_uri('http://mlflow-server:5000')
    mlflow.set_experiment('experiment_covertype-1')
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "linear_regression")
        mlflow.sklearn.log_model(model, "linear_regression_model")
    print("Model trained and logged to MLflow.")

dag_train_model_mlflow = DAG(
    'train_model_mlflow',
    description='Train model and log to MLflow',
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False
)

task_train_model = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model_and_log_to_mlflow,
    dag=dag_train_model_mlflow
)