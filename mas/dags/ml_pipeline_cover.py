from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

# Definir los argumentos del DAG
default_args = {
    'owner': 'Oscar C',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['oecorrechag@gmail.com'],
    'retries': 1,
    'start_date': datetime(2024, 5, 20),
}

# Función para cargar el dataset iris
def load_iris_dataset():
    df = pd.read_csv('https://raw.githubusercontent.com/oecorrechag/MLOps/main/testeos/mlflow_fast/covertype_train.csv?token=GHSAT0AAAAAACR4QJCZGXNAE43TTASEOD4AZSRHNSA', sep = ',', decimal = '.', header = 0, encoding = 'utf-8')
    df = df.rename(columns=lambda x: x.replace(' ', '_').lower())
    df = df.loc[:,['elevation', 'horizontal_distance_to_roadways', 'hillshade_9am', 
                'horizontal_distance_to_fire_points', 'cover_type']]
    print('cargo los datos')
    return df

# Función para entrenar el modelo
def train_model():
    df = pd.read_csv('https://raw.githubusercontent.com/oecorrechag/MLOps/main/testeos/mlflow_fast/covertype_train.csv?token=GHSAT0AAAAAACR4QJCZGXNAE43TTASEOD4AZSRHNSA', sep = ',', decimal = '.', header = 0, encoding = 'utf-8')
    df = df.rename(columns=lambda x: x.replace(' ', '_').lower())
    df = df.loc[:,['elevation', 'horizontal_distance_to_roadways', 'hillshade_9am', 
                'horizontal_distance_to_fire_points', 'cover_type']]
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('cover_type', axis = 1), df['cover_type'], test_size=0.8, random_state=42)

    model =  DecisionTreeClassifier()
    
    model.fit(X_train, y_train)  # Train model
    predictions = model.predict(X_test)  # Predictions

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')  
    print('modelo entrenado')

# Definir el DAG
dag = DAG(
    'covertype_classification_dag',
    default_args=default_args,
    description='Un DAG para cargar el conjunto de datos covertype, entrenar un modelo y guardar los resultados en un archivo CSV.',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
load_data = PythonOperator(task_id='load_data', python_callable=load_iris_dataset, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
# save_results = PythonOperator(task_id='save_to_csv', python_callable=save_to_csv, provide_context=True, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> load_data >> train >> end_operator
