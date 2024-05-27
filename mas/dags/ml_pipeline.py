from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Función para cargar el dataset iris
def load_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df

# Función para entrenar el modelo
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

# Función para guardar los resultados en un archivo CSV
def save_to_csv():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(iris.data)
    df = pd.DataFrame({
        'sepal_length': iris.data[:, 0],
        'sepal_width': iris.data[:, 1],
        'petal_length': iris.data[:, 2],
        'petal_width': iris.data[:, 3],
        'species': iris.target,
        'predicted_species': predictions
    })
    df.to_csv('results.csv', index=False)

# Definir los argumentos del DAG
default_args = {
    'owner': 'Oscar C',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['oecorrechag@gmail.com'],
    'retries': 1,
    'start_date': datetime(2024, 5, 20),
}


# Definir el DAG
dag = DAG(
    'iris_classification_dag',
    default_args=default_args,
    description='Un DAG para cargar el conjunto de datos iris, entrenar un modelo y guardar los resultados en un archivo CSV.',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
load_data = PythonOperator(task_id='load_data', python_callable=load_iris_dataset, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
save_results = PythonOperator(task_id='save_to_csv', python_callable=save_to_csv, provide_context=True, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> load_data >> train >> save_results >> end_operator
