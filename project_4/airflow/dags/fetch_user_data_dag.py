from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def fetch_data_and_store():
    import requests
    from sqlalchemy import create_engine, inspect
    from pandas import DataFrame

    # Conn
    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://user1:password1@mysql1/database1')

    # get data from API
    GROUP_NUMBER = 1
    endpoint = f"http://10.43.101.149/data?group_number={GROUP_NUMBER}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        g = data["group_number"]
        b = data["batch_number"]
        data = data["data"]
        df = DataFrame(data)
        df.columns = ['brokered_by','status','price','bed','bath','acre_lot','street','city','state',
                      'zip_code','house_size','prev_sold_date']
        data_collected = True
        print("Data collected")
    else:
        data_collected = False
        print(f"Error retrieving data: {response.status_code}")
    
    # Save data into raw_data
    # Check if table exists
    inspector = inspect(mysql_engine)
    if not inspector.has_table('raw_data_bienes_raices_user') and data_collected:
        df.to_sql('raw_data_bienes_raices_user', con=mysql_engine, if_exists='replace', index=False)
    elif inspector.has_table('raw_data_bienes_raices_user') and data_collected:
        df.to_sql('raw_data_bienes_raices_user', con=mysql_engine, if_exists='replace', index=False)
    print("Your data has been successfully stored")

# args dag
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False,
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# build DAG
with DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG para la ingesta de datos desde una API a MySQL',
    schedule_interval='0 0 * * 1',  # every monday 
    catchup=False,
) as dag:
    
    # task definition
    ingest_data_task = PythonOperator(
        task_id='ingest_data',
        python_callable=fetch_data_and_store,
        provide_context=True
    )

    ingest_data_task
