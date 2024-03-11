from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect
from dags.trainig import training

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'start_date': datetime(2024, 3, 10),
    'schedule_interval': None,
}

with DAG(dag_id='ml_models',
         default_args=default_args,
         description='DAG to create and store ML models',
         schedule_interval=None) as dag:
       
       enter_point = DummyOperator(
              task_id='enter_point')

       t3 = PythonOperator(task_id='ml_models', 
                           python_callable=training)
    
       enter_point >> t3
