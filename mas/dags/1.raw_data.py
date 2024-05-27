import os
import pandas as pd
import requests
from datetime import datetime
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

# Definir los argumentos del DAG
default_args = {
    'owner': 'Group 1',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['xxxxx@gmail.com'],
    'retries': 1,
    'start_date': datetime(2024, 5, 20),
}

# Función para cargar el dataset iris
def raw_data(group_number):

    engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')

    url = f"http://10.43.101.149/data?group_number={group_number}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        g = data["group_number"]
        b = data["batch_number"]
        data = data["data"]
        df = pd.DataFrame(data)
        df.columns = ['brokered_by','status','price','bed','bath','acre_lot','street','city','state',
                      'zip_code','house_size','prev_sold_date']
        
        # Guardar los datos en MySQL
        df.to_sql('raw_data', con=engine, if_exists='append', index=False)
        return print("Datos raw_data guardados en MySQL") 
    else:
        print(f"Error retrieving data: {response.status_code}")
        return None


# Definir el DAG
dag = DAG(
    'get_data_dag',
    default_args=default_args,
    description='En este dag se cargan los datos raw',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
raw_data = PythonOperator(task_id='raw_data', python_callable=raw_data, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> raw_data >> end_operator
