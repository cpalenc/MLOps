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
import xgboost as xgb
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    dag_id = '3.Entrenamiento_de_modeloV2',
    default_args=default_args,
    description='A Machine Learning workflow for price house prediction',
    schedule_interval=None,
    start_date=days_ago(1)
    #tags=['ml', 'diabetes'],
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
    df = df.dropna()


    # Separar características (X) y variable objetivo (y)
    X = df.drop(['price'], axis=1)
    y = df['price']





    ## Dividir conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("house_price_1")
    mlflow.autolog(log_input_examples=True, log_model_signatures=True)

    current_experiment = dict(mlflow.get_experiment_by_name('house_price_1'))
    experiment_id = current_experiment['experiment_id']

    # Modelo Decision Tree
    model_name = 'XGBoost'
    RUN_NAME = f'House price Classifier Experiment {model_name}'
   # params = {'max_depth': 3, 'min_samples_split': 2}
    params = {
                'xgb__colsample_bytree': 0.8123620356542087, 'xgb__learning_rate': 0.2952142919229748,
                'xgb__max_depth': 5, 'xgb__n_estimators': 121, 'xgb__subsample': 0.8795975452591109
            }
    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME):
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_params(params)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        
        
            
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        
        
        
        mlflow.set_tag("Training Info", f"{model_name} model")
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=f"price_{model_name}_model",
                                              signature=signature, input_example=X_train,
                                              registered_model_name=f"tracking-price-{model_name}")
        mlflow.end_run() 

        client = MlflowClient()
        client.set_registered_model_tag(f"tracking-price-{model_name}", "task", "Regression")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

preprocess_data_task
