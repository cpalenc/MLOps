from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor

    import os
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("mlflow-tracking-diabetes")

    # this is the magical stuff
    mlflow.autolog(log_input_examples=True, log_model_signatures=True)

    # train the model
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)
        
    
    
    
with DAG(dag_id="track_metrics",
         description="Utilizando Python Operator",
         schedule_interval="@once",
         start_date=datetime(2024,5,5)
 ) as dag:
    
    t1 = PythonOperator(task_id= "hello_with_python",
                        python_callable=print_hello)