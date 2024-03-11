import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, inspect

np.random.seed(42)

def training():
    """
    This function loads the data, builds, trains, and saves the model.
    Args:
        None
    Returns:
        None
    """
    df = _get_data()
    model = _build_model(df)


def _get_data():
    """
    Connects to the MySQL database 'penguin_data' and retrieves all data
    from the 'penguins' table.

    Handles potential errors during connection or data retrieval.

    Returns:
        pandas.DataFrame: The retrieved data from the 'penguins' table.
        None: If an error occurs during connection or data retrieval.
    """
    try:
        # Connect to MySQL using secure credentials (avoid hardcoding)
        engine = create_engine('mysql://root:airflow@mysql:3306/penguin_data')

        # Read data from the 'penguins' table
        df = pd.read_sql("SELECT * FROM penguins", con=engine)

        return df

    except Exception as e:
        print(f"An error occurred while getting data: {e}")
        return None
    

def _build_model(df: pd.DataFrame):
    """
    Builds and saves machine learning models for penguin species classification.

    Performs data preprocessing steps including:
        - Column name cleaning
        - Handling categorical features (label encoding, one-hot encoding)
        - Imputation for missing values
        - Feature scaling

    Creates and trains two models:
        - Decision Tree Classifier
        - Logistic Regression

    Saves the trained models using pickle.

    Args:
        df (pd.DataFrame): The input DataFrame containing penguin data.

    Returns:
        None
    """

    df = df.rename(columns=lambda x: x.replace(' ', '_').lower())
    df.rename(columns={'culmen_length_(mm)':'culmen_length', 
                       'culmen_depth_(mm)':'culmen_depth',
                       'flipper_length_(mm)':'flipper_length',
                       'body_mass_(g)':'body_mass',
                       'delta_15_n_(o/oo)':'delta_15',
                       'delta_13_c_(o/oo)':'delta_13'
                       }, inplace=True)
    df = df.drop(['region', 'stage'], axis=1)

    ## Variable salida
    df["species"] = LabelEncoder().fit_transform(df.species)

    # Otras variables label
    df["individual_id"] = LabelEncoder().fit_transform(df.individual_id)
    df["clutch_completion"] = LabelEncoder().fit_transform(df.clutch_completion)
    df["date_egg"] = LabelEncoder().fit_transform(df.date_egg)
    df["sex"] = LabelEncoder().fit_transform(df.sex)
    df["comments"] = LabelEncoder().fit_transform(df.comments)

    # ONE
    one_hot_encoded = pd.get_dummies(df['island'], prefix='island', drop_first=True)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df["island_Dream"] = LabelEncoder().fit_transform(df.island_Dream)
    df["island_Torgersen"] = LabelEncoder().fit_transform(df.island_Torgersen)

    one_hot_encoded = pd.get_dummies(df['studyname'], prefix='studyname', drop_first=True)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df["studyname_PAL0809"] = LabelEncoder().fit_transform(df.studyname_PAL0809)
    df["studyname_PAL0910"] = LabelEncoder().fit_transform(df.studyname_PAL0910)

    df = df.drop(['studyname','island'], axis=1)

    ### imputar datos
    imputer = KNNImputer(n_neighbors = 3)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Separar las características (X) y la variable objetivo (y)
    X = df[['culmen_length','flipper_length','culmen_depth']]
    y = df['species']

    # Escalar las características
    scaler = MinMaxScaler()
    scaler2 = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = scaler2.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=420)

    dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    lr = LogisticRegression(random_state=42).fit(X_train, y_train)

    pickle_out = open("models/cl_dt.pkl","wb")
    pickle.dump(dt, pickle_out)
    pickle_out.close()

    pickle_out = open("models/cl_lr.pkl","wb")
    pickle.dump(lr, pickle_out)
    pickle_out.close()
