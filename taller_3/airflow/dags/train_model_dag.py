from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
#import joblib

# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://airflow:airflow@mysql/airflow')

# Define function to train the model
def train_species_prediction_model(X, y):
    # Training the linear regression model
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add column of ones for bias
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Function to load data from MySQL database
def load_data_from_mysql():
    # Query data from MySQL database
    query = "SELECT * FROM penguins_data"
    df = pd.read_sql(query, con=mysql_engine)
    return df.dropna()

# Function to preprocess data
def preprocess_data(df):
    X = df.drop(columns=['Species'])
    y = df['Species']
    return X, y

# Function to train the model
def train_model():
    df = load_data_from_mysql()
    X, y = preprocess_data(df)
    theta = train_species_prediction_model(X, y)
    # Save the trained model
    #joblib.dump(theta, '/opt/airflow/data/trained_model.pkl')
    #print("Model trained and saved successfully!")

# Define the DAG
train_model_dag = DAG(
    'train_model',
    description='Train model using data from the database',
    schedule_interval=None,  # Don't schedule automatically, trigger manually
    start_date=datetime(2024, 3, 12),  # Start date of execution
    catchup=False  # Avoid running previous tasks if execution is delayed
)

# Define the task to train the model
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=train_model_dag
)
