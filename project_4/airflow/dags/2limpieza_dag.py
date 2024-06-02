from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)

# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://user1:password1@mysql1/database1')

def import_raw_data():
    
    query = "SELECT * FROM raw_data_bienes_raices;"
    df = pd.read_sql(query, con=mysql_engine)
    return df
    
def transform_data(df):
    
    df = df.copy()
    df = df.loc[:,['status','price','bed','bath','acre_lot','state','prev_sold_date','house_size']]
    df = df.dropna()
    df["año"] = pd.to_datetime(df['prev_sold_date']).dt.year
    df["decada"] = (df["año"] // 10) * 10
    df['status'] = df['status'].replace({'for_sale':0 , 'sold':1})
    
    df = df[df['bed'] < 7]
    df = df[df['bath'] < 5]
    df = df[df['price'] < 300000]
    df = df[df['acre_lot'] <= 0.0894211]
    df = df[df['house_size'] < 3500]
    df = df[df['decada'] >= 1980]
    
    state_dict = {
            'alabama': 1, 'alaska': 2, 'arizona': 3, 'arkansas': 4, 'california': 5, 'colorado': 6,
            'connecticut': 7, 'delaware': 8, 'florida': 9, 'georgia': 10, 'hawaii': 11, 'idaho': 12,
            'illinois': 13, 'indiana': 14, 'iowa': 15, 'kansas': 16, 'kentucky': 17, 'louisiana': 18,
            'maine': 19, 'maryland': 20, 'massachusetts': 21, 'michigan': 22, 'minnesota': 23,
            'mississippi': 24, 'missouri': 25, 'montana': 26, 'nebraska': 27, 'nevada': 28,
            'nueva hampshire': 29, 'nueva jersey': 30, 'nueva york': 31, 'nuevo mexico': 32,
            'carolina del norte': 33, 'dakota del norte': 34, 'ohio': 35, 'oklahoma': 36, 'oregon': 37,
            'pensilvania': 38, 'rhode island': 39, 'carolina del sur': 40, 'dakota del sur': 41,
            'tennessee': 42, 'texas': 43, 'utah': 44, 'vermont': 45, 'virginia': 46, 'washington': 47,
            'virginia occidental': 48, 'wisconsin': 49, 'wyoming': 50
                }
    
    df['state'] = df['state'].str.lower()
    df['state'] = df['state'].map(state_dict)
    
    
    #isolation_forest.fit(df.loc[:,['price', 'bed', 'bath', 'acre_lot', 'state', 'house_size']])
    #anomalies = isolation_forest.predict(df.loc[:,['price', 'bed', 'bath', 'acre_lot', 'state', 'house_size']])
    #df = df[anomalies == 1]
    
    df = df.loc[:,['price', 'bed', 'bath', 'acre_lot', 'state', 'house_size']]  
    df = df.dropna(how = 'all')  
    return df




def load_to_database(df):
    
    NEW_DB_CONN_ID = 'new_db_conn'
    new_db_engine = create_engine('mysql://user2:password2@mysql2/database2')
    

    inspector = inspect(new_db_engine)
    if not inspector.has_table('clean_data'):
        df.head(0).to_sql('clean_data', con=new_db_engine, if_exists='append', index=False)  # Create empty table
        df.to_sql('clean_data', con=new_db_engine, if_exists='append', index=False)
    else:
        df.to_sql('clean_data', con=new_db_engine, if_exists='append', index=False)       

    
def main():
    load_to_database(transform_data(import_raw_data()))



with DAG (dag_id= "2.Limpieza_y_carga_a_db2",
          description= "transform raw data and load to clean data",
          schedule_interval= "@once",
          start_date= datetime(2024,5,27)) as dag:
        
            t1 = PythonOperator (task_id="loadData",
                                python_callable=main)