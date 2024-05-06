from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
import os

def tranformaciones():
    
    MYSQL_CONN_ID = 'mysql_conn'
    mysql_engine = create_engine('mysql://user2:password2@mysql2/database2')
    # Load data from MySQL
    query = "SELECT * FROM clean_data;"
    df = pd.read_sql(query, con=mysql_engine)

    df = df.loc[:,['age', 'discharge_disposition_id', 'time_in_hospital', 'num_lab_procedures', 
                   'num_procedures', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 
                   'readmitted']]
    
    # Separar características (X) y variable objetivo (y)
    X = df.drop(['readmitted'], axis=1)
    y = df['readmitted']

    ## Escalar características
    scaler = MinMaxScaler()
    scaler2 = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = scaler2.fit_transform(X_scaled)

    ## Balanceo de clases
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    ## Dividir conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://Mlflow:5000")
    EXPERIMENT_NAME = "Readmitted-Survived-Classifier-Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)
    current_experiment = dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    experiment_id = current_experiment['experiment_id']

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://Minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # Modelo Decision Tree
    model_name = 'Decision Tree'
    RUN_NAME = f'Readmitted Classifier Experiment {model_name}'
    params = {'max_depth': 3, 'min_samples_split': 2}
    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME):
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')  
        mlflow.log_params(params)
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.log_metric(f"{model_name}_f1", f1)
        mlflow.set_tag("Training Info", f"{model_name} model for Readmitted")
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=f"readmitted_{model_name}_model",
                                              signature=signature, input_example=X_train,
                                              registered_model_name=f"tracking-readmitted-{model_name}")
        mlflow.end_run() 

    client = MlflowClient()
    client.set_registered_model_tag("tracking-readmitted-Decision Tree", "task", "classification")

# Define los argumentos del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 6),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define el DAG
dag = DAG('readmitted_classifier_dag',
          default_args=default_args,
          schedule_interval='@daily')

# Define la tarea Python para ejecutar las transformaciones
transform_task = PythonOperator(
    task_id='transform_task',
    python_callable=tranformaciones,
    dag=dag,
)

# Establece la secuencia de tareas en el DAG
transform_task
