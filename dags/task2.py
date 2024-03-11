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

def delete_mysql():
        
    # Connecting to MySQL and deleting data from tables.
    engine = create_engine('mysql://root:airflow@mysql:3306/penguin_data')
    
    # Utilize an "inspect" object; if it contains information, it indicates the existence of the 
    # table; otherwise, it signifies that the table has not been created yet.
    inspector = inspect(engine)
    if 'penguins' in inspector.get_table_names():
        engine.execute("DROP TABLE penguins")
        print("The data has been successfully deleted from MySQL.")
    else:
        print("The <penguins> table is not present in the database table collection from MySQL")


with DAG(dag_id = 'delete_data_dag',
         default_args=default_args,
         description='DAG to delete content from a MySQL database',
         schedule_interval=None) as dag:
     
       t2 = PythonOperator(task_id='delete_mysql', 
                           python_callable=delete_mysql)
    