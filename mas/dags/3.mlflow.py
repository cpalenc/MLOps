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
def make_model(group_number):

    engine = create_engine('mysql+pymysql://root:airflow@mysql:3306/db')

    url = f"http://10.43.101.149/data?group_number={group_number}"
    response = requests.get(url)

    # Consulta para cargar los datos desde la tabla en la base de datos
    query = "SELECT * FROM raw_data"
    # Leer los datos desde MySQL
    df = pd.read_sql(query, con=engine)


    # Eliminar los registros con faltantes
    df = df.dropna()
    # Convertir en string el zip code
    df['zip_code'] = df['zip_code'].astype(str)
    

    # Guardar los datos en MySQL
    df.to_sql('clean_data', con=engine, if_exists='append', index=False)

    return print("Datos clean_data guardados en MySQL") 

# Definir el DAG
dag = DAG(
    'clean_data_dag',
    default_args=default_args,
    description='En este dag se cargan los datos raw',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
clean_data = PythonOperator(task_id='clean_data', python_callable=clean_data, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> clean_data >> end_operator
