from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'start_date': datetime(2024, 3, 10),
    'schedule_interval': None,
}

def load_mysql():
    penguins = pd.read_csv("/opt/airflow/data/penguins_lter.csv")

    # To establish a connection and save data into MySQL tables.
    engine = create_engine('mysql://root:airflow@mysql:3306/penguin_data')
    penguins.to_sql('penguins', con=engine, index=False, if_exists='replace')
    print("The data has been successfully loaded into MySQL.")

with DAG(dag_id = 'load_data', 
         default_args=default_args, 
         description='DAG to connect and save data to MySQL',
         schedule_interval=None) as dag:
       
       t1 = PythonOperator(task_id='load_mysql',
                           python_callable=load_mysql)
     