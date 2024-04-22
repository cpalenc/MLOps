import os
import requests
from json import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Fetch data from API and save locally
api_url = "http://10.43.101.149/data?group_number=1"
response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()
    with open("api/data/covertype.json", "w") as outfile:
        dump(data, outfile)
    print("Data fetched from API and saved locally.")
else:
    print("Error fetching data from API.")

# Load data from JSON into DataFrame
with open('./data/covertype.json') as f:
    data = load(f)
df = pd.DataFrame(data['data'])
# Assign column names
column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type',
                'Cover_Type']
df.columns = column_names

# Set target and input values
y = df['Cover_Type']
df.drop('Cover_Type', axis=1, inplace=True)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Column transformation and pipeline
column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                        ["Wilderness_Area", "Soil_Type"]),
                                      remainder='passthrough')

pipe = Pipeline(steps=[("column_trans", column_trans),
                       ("scaler", StandardScaler(with_mean=False)),
                       ("RandomForestClassifier", RandomForestClassifier())])

param_grid = {'RandomForestClassifier__max_depth': [1, 2, 3, 10],
              'RandomForestClassifier__n_estimators': [10, 11]}
search = GridSearchCV(pipe, param_grid, n_jobs=2)

# MLflow setup and autologging
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://0.0.0.0:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("mlflow_tracking_examples")

mlflow.autolog(log_model_signatures=True, log_input_examples=True)

with mlflow.start_run(run_name="autolog_with_pipeline") as run:
    search.fit(X_train, y_train)

# Retrieve and use the trained model
model_name = "modelo1"
mlflow.set_tracking_uri("http://localhost:5000")
model_production_uri = f"models:/{model_name}/production"
loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)

# Example prediction
example_test = X_test.iloc[0].to_frame().T
print('real: ', y_test.iloc[0])
print('prediction: ', loaded_model.predict(example_test))