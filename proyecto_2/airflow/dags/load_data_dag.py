from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect

# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://airflow:airflow@mysql/airflow')

# Function to load data into MySQL
def load_data_to_mysql():
    data = pd.read_json("/opt/airflow/data/covertype.json")
    
    # Check if table exists
    inspector = inspect(mysql_engine)
    if not inspector.has_table('dataset_covertype'):
        data.head(0).to_sql('dataset_covertype', con=mysql_engine, if_exists='replace', index=False)  # Create empty table
    else:
        data.to_sql('dataset_covertype', con=mysql_engine, if_exists='replace', index=False)

# Define the DAG
load_covertype_data_to_mysql_dag = DAG(
    'load_covertype_data_to_mysql',
    description='load covertype data to mysql',
    schedule_interval=None,  # Don't schedule automatically, trigger manually
    start_date=datetime(2024, 4, 10),  # Start date of execution
    catchup=False  # Avoid running previous tasks if execution is delayed
)

# Define the task to load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_to_mysql,
    dag=load_covertype_data_to_mysql_dag
)
