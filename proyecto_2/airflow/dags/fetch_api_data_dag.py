from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import requests
import json

def fetch_data_and_save_to_json():
    api_url = "http://10.43.101.149/data?group_number=1"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        with open("/opt/airflow/data/covertype.json", "w") as outfile:
            json.dump(data, outfile)
        print("Data fetched from API and saved locally.")
    else:
        print("Error fetching data from API.")

dag_fetch_api_data = DAG(
    'fetch_api_data',
    description='Fetch data from API and save locally',
    schedule_interval=None, #timedelta(minutes=5),  # Ejecutar cada 5 minutos
    start_date=datetime(2024, 4, 10),
    catchup=False
)

task_fetch_data = PythonOperator(
    task_id='fetch_data_task',
    python_callable=fetch_data_and_save_to_json,
    dag=dag_fetch_api_data
)
