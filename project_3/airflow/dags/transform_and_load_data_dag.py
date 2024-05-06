from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect



# Define MySQL connection
MYSQL_CONN_ID = 'mysql_conn'
mysql_engine = create_engine('mysql://user1:password1@mysql1/database1')

def import_raw_data():
    # Load data from MySQL
    query = "SELECT * FROM diabetes_data;"
    df = pd.read_sql(query, con=mysql_engine)
    return df
    
def transform_data(df):
        cols_to_remove = ['encounter_id', 'patient_nbr',  'payer_code',  'examide', 'citoglipton','medical_specialty']
        df = df.drop(columns=cols_to_remove, axis = 1) 

        df = df.replace('?',np.nan)
        df['gender'] = df['gender'].replace({'Female':1,
                                            'Male':0,
                                            'Unknown/Invalid' : np.nan})

        df['age'] = df['age'].replace({'[0-10)':5,
                                    '[10-20)':15, 
                                    '[20-30)':25, 
                                    '[30-40)':35,
                                    '[40-50)':45, 
                                    '[50-60)':55,
                                    '[60-70)':65, 
                                    '[70-80)':75,
                                    '[80-90)':85, 
                                    '[90-100)':95})

        df['race'] = df['race'].replace({'Caucasian':0,
                                        'AfricanAmerican':1,
                                        'Other':2, 
                                        'Asian':3, 
                                        'Hispanic':4,
                                        '?': np.nan})

        df['readmitted'] = df['readmitted'].replace({'NO':0,
                                                    '>30':1,
                                                    '<30':0})

        df['weight'] = df['weight'].replace({'[0-25)':12.5,
                                            '[25-50)': 37.5,
                                            '[50-75)': 62.5,
                                            '[75-100)': 87.5,
                                            '[100-125)': 112.5,
                                            '[125-150)': 137.5, 
                                            '[150-175)': 162.5, 
                                            '[175-200)': 187.5, 
                                            '>200': 200,
                                            '?': np.nan})

        df['change'] = df['change'].replace({'Ch':1,
                                            'No':0})

        df['diabetesMed'] = df['diabetesMed'].replace({'Yes':1,
                                                    'No':0})
                                                
        similar_columns = ['metformin', 'repaglinide','nateglinide', 'chlorpropamide', 'glimepiride','acetohexamide', 
                            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                            'miglitol','troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

        for col in similar_columns:
            df[col] = df[col].replace({'No':-1,'Steady':0,'Up':1,'Down':-1})
            
        
        df['A1Cresult'] = df['A1Cresult'].replace({'>7': 1,
                                                '>8': 1,
                                                'Norm': 0,
                                                'None': -1})

        df['max_glu_serum'] = df['max_glu_serum'].replace({'>200': 1,
                                                        '>300': 1,
                                                        'Norm': 0,
                                                        'None': -1})   

        diag_columns = ['diag_1', 'diag_2', 'diag_3']

        for col in diag_columns:
            df.loc[df[col].str.contains('V|E', na=False), col] = 0
            df[col] = df[col].astype(float)
            df[col] = df[col] // 100
            
            
        df = df.dropna()
        
        return df




def load_to_database(df):
    # actualizar datos de conexión
    NEW_DB_CONN_ID = 'new_db_conn'
    new_db_engine = create_engine('mysql://user2:password2@mysql2/database2')
    
    #df.to_sql('clean_data', con=new_db_engine, index=False, if_exists='replace')
    
     # Check if table exists
    inspector = inspect(new_db_engine)
    if not inspector.has_table('clean_data'):
        df.head(0).to_sql('clean_data', con=new_db_engine, if_exists='replace', index=False)  # Create empty table
        df.to_sql('clean_data', con=new_db_engine, if_exists='replace', index=False)
    else:
        df.to_sql('clean_data', con=new_db_engine, if_exists='replace', index=False)       

    
def main():
    load_to_database(transform_data(import_raw_data()))




with DAG (dag_id= "transform_data",
          description="transform data and load to MySQL",
          schedule_interval="@once",
          start_date=datetime (2024,5,4)) as dag:
        
            t1 = PythonOperator (task_id="loadData",
                                python_callable=main)
           