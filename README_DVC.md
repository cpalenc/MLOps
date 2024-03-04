# How to use DVC
## Using with gogle cloud:
### Install DVC
```bash
pip install dvc dvc-gs
```
### Create storage in gc

#### create credencial 
<img src="image.png" alt="create credential" style="width:500px;height:500px;">

#### create credential as service account 
<img src="image-1.png" alt="type credential" style="width:500px;height:500px;">

#### create credencial with rol cloud storage as admiin rol 
<img src="image-2.png" alt="rol credential" style="width:500px;height:500px;">

#### create key and save as json into project 
<img src="image-3.png" alt="rol credential" style="width:500px;height:500px;">

#### create bucket and 2 folders <dataset> and <model>
<img src="image-4.png" alt="rol credential" style="width:900px;height:500px;">


### conect to storage from project
## create enviroment variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS=$(realpath ./mlops-puj-cbd9b5b5e4ac.json)
```
## connect to dataset folder
```bash
dvc init
dvc remote add dataset-track gs://model-dataset-tracker-ccps/dataset
```

### connect to dataset folder
```bash
dvc remote add dataset-track gs://model-dataset-tracker-ccps/dataset
```
### add files
```bash
dvc add proyecto_1/data/covertype/covertype_train.csv --to-remote -r dataset-track
```

## connect to model folder
```bash
dvc remote add model-track gs://model-dataset-tracker-ccps/model
```
### add model to track
```bash
dvc add model/model.pkl --to-remote -r model-track
```


