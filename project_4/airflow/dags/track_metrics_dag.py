from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'diabetes_ml_workflow',
    default_args=default_args,
    description='A Machine Learning workflow for diabetes prediction',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['ml', 'diabetes'],
)

def preprocess_data():
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

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

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("diabetes-1")
    mlflow.autolog(log_input_examples=True, log_model_signatures=True)

    current_experiment = dict(mlflow.get_experiment_by_name('diabetes-1'))
    experiment_id = current_experiment['experiment_id']

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

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

preprocess_data_task
