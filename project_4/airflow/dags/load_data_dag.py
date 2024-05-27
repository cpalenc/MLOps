from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine, inspect
from ucimlrepo import fetch_ucirepo

# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://user1:password1@mysql1/database1')

# Function to fetch dataset and load data into MySQL
def fetch_and_load_data():
    import os
    import requests
    import pandas as pd
    _data_root = './data'
    # Path to the raw training data
    _data_filepath = os.path.join(_data_root, 'Diabetes.csv')
    # Download data
    os.makedirs(_data_root, exist_ok=True)
    if not os.path.isfile(_data_filepath):
        #https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
        url = 'https://docs.google.com/uc?export= \
        download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        r = requests.get(url, allow_redirects=True, stream=True)
        open(_data_filepath, 'wb').write(r.content)
        
    data = pd.read_csv("./data/Diabetes.csv")
    
    # Check if table exists
    inspector = inspect(mysql_engine)
    if not inspector.has_table('diabetes_data'):
        data.head(0).to_sql('diabetes_data', con=mysql_engine, if_exists='replace', index=False)  # Create empty table
        data.to_sql('diabetes_data', con=mysql_engine, if_exists='replace', index=False)
    else:
        data.to_sql('diabetes_data', con=mysql_engine, if_exists='replace', index=False)

# Define the DAG
dag = DAG(
    'load_diabetes_data_to_mysql',
    description='Load diabetes data into MySQL',
    schedule_interval=None,  # Don't schedule automatically, trigger manually
    start_date=datetime(2024, 5, 5),  # Start date of execution
    catchup=False  # Avoid running previous tasks if execution is delayed
)

# Define the task to fetch and load data
fetch_and_load_data_task = PythonOperator(
    task_id='fetch_and_load_diabetes_data_task',
    python_callable=fetch_and_load_data,
    dag=dag
)
