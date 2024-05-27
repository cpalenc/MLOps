from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow

# conectar con mlflow y minio
from mlflow.models import infer_signature
mlflow.set_tracking_uri("http://Mlflow:5000")

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://Minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

# Definir los argumentos del DAG
default_args = {
    'owner': 'Oscar C',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['oecorrechag@gmail.com'],
    'retries': 1,
    'start_date': datetime(2024, 5, 20),
}

def entre_save():

    """Aca solo cargare los datos y colocare el modelo en mlflow"""

    df = pd.read_csv('https://raw.githubusercontent.com/oecorrechag/MLOps/main/testeos/mlflow_fast/covertype_train.csv?token=GHSAT0AAAAAACR4QJCZGXNAE43TTASEOD4AZSRHNSA', sep = ',', decimal = '.', header = 0, encoding = 'utf-8')
    df = df.rename(columns=lambda x: x.replace(' ', '_').lower())
    df = df.loc[:,['elevation', 'horizontal_distance_to_roadways', 'hillshade_9am', 
                'horizontal_distance_to_fire_points', 'cover_type']]

    print('cargo los datos')

    X_train, X_test, y_train, y_test = train_test_split(df.drop('cover_type', axis = 1), df['cover_type'], test_size=0.8, random_state=42)

    EXPERIMENT_NAME = "Cover-Survived-Classifier-Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)

    current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    experiment_id=current_experiment['experiment_id']

    print('inicia el experimento')

    model_name = 'Decision Tree'
    RUN_NAME = f'Cover Classifier Experiment {model_name}'
    params = {'max_depth':3, 'min_samples_split':2}
    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME):
        
        model =  DecisionTreeClassifier(**params)
        
        model.fit(X_train, y_train)  # Train model
        predictions = model.predict(X_test)  # Predictions

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')  
        
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.log_metric(f"{model_name}_f1", f1)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"{model_name} model for Cover")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        #log the model

        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f"cover_{model_name}_model",
        signature=signature,
        input_example=X_train,
        registered_model_name=f"tracking-cover-{model_name}",)

        print('finaliza el experimento')

        mlflow.end_run() 

# Definir el DAG
dag = DAG(
    'covertype_classification_dag_old',
    default_args=default_args,
    description='Un DAG para Carga y entrenamiento de covertype.',
    schedule_interval=None
)

# Definir los operadores
start_operator = DummyOperator(task_id='start', dag=dag)
entre_save = PythonOperator(task_id='load_data', python_callable=entre_save, dag=dag)
end_operator = DummyOperator(task_id='end', dag=dag)

# Definir el flujo del DAG
start_operator >> entre_save >>  end_operator
