from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine

# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://user1:password1@mysql/database1')

# Function to clear database content
def clear_database_content():
    # Connect to the database
    with mysql_engine.connect() as conn:
        # Execute SQL query to delete content from table
        conn.execute('DELETE FROM diabetes_data')

# Define the DAG
clear_database_content_dag = DAG(
    'clear_database_content',
    description='Clear content from the database',
    schedule_interval=None,  # Don't schedule automatically, trigger manually
    start_date=datetime(2024, 5, 5),  # Start date of execution
    catchup=False  # Avoid running previous tasks if execution is delayed
)

# Define the task to clear database content
clear_database_task = PythonOperator(
    task_id='clear_database_task',
    python_callable=clear_database_content,
    dag=clear_database_content_dag
)