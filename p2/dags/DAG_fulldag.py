from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import mlflow
import requests
from json import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define default arguments for the DAG
default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 22),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Instantiate the DAG
dag = DAG('ml_pipeline_dag', default_args=default_args, description='Machine Learning Pipeline DAG', schedule_interval=None)

def preprocess_data():
    with open('./api/data/covertype.json') as f:
        data = load(f)
    df = pd.DataFrame(data['data'])
    column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type',
                    'Cover_Type']
    df.columns = column_names
    y = df['Cover_Type']
    df.drop('Cover_Type', axis=1, inplace=True)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                            ["Wilderness_Area", "Soil_Type"]),
                                          remainder='passthrough')
    pipe = Pipeline(steps=[("column_trans", column_trans), ("scaler", StandardScaler(with_mean=False)),
                            ("RandomForestClassifier", RandomForestClassifier())])
    param_grid = {'RandomForestClassifier__max_depth': [1, 2, 3, 10], 'RandomForestClassifier__n_estimators': [10, 11]}
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X_train, y_train)
    return search

def log_to_mlflow():
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    mlflow.set_experiment("mlflow_tracking_examples")
    with mlflow.start_run(run_name="autolog_with_pipeline") as run:
        mlflow.sklearn.log_model(search, "RandomForestModel")

# Define tasks for the DAG
preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

log_mlflow_task = PythonOperator(
    task_id='log_to_mlflow',
    python_callable=log_to_mlflow,
    dag=dag,
)

# Define task dependencies
preprocess_data_task >> train_model_task >> log_mlflow_task
