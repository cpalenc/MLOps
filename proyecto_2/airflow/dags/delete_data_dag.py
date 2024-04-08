from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine, inspect

def drop_table_if_exists():
    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://airflow:airflow@mysql/airflow')
    inspector = inspect(mysql_engine)
    if inspector.has_table('dataset_table'):
        with mysql_engine.connect() as connection:
            connection.execute('DROP TABLE dataset_covertype')
        print("Table dataset_table dropped.")
    else:
        print("Table dataset_table does not exist.")

dag_drop_table_if_exists = DAG(
    'drop_table_if_exists',
    description='Drop table if exists in database',
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False
)

task_drop_table = PythonOperator(
    task_id='drop_table_task',
    python_callable=drop_table_if_exists,
    dag=dag_drop_table_if_exists
)
