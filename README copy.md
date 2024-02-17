# MLOps

"MLOps" was created to document all topics and development covered in the MLOps class. Written in Python 3.

## Authors

- Oscar Correa - [@oecorrechag](https://github.com/oecorrechag)
- Daniel Chavarro - [@anielFchavarro](https://github.com/anielFchavarro)
- Cristhian Palencia - [@cpalenc](https://github.com/cpalenc)



## **Firsts steps, before clone the repo!**

### How Configure in the terminal and clone repository

remenber config your local git

```Bash
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com" 
git clone https://github.com/oecorrechag/MLOps.git
 ```


## **Next, How to start in this Repo**

### Prepare the environment
```bash
# Ubuntu  
sudo apt install python3.10 -y 

# Fedora
sudo dnf install python3.10 -y

# Download URL: https://www.python.org/downloads/release/python-31013/
```

### Install requirements
Create virtual environment and install dependencies.

```bash
# create virtual enviroment
python3 -m venv <venv_name> 
```

```bash
# activate virtual enviroment
source  <venv_name>/bin/activate
```
```bash
# install and update all requirements
pip install -r requirements.txt
```
## Run app
remember: you must be in the directory where the main.py file is located "level_0/app"

### Run app in local server from docker image
create docker image
```bash
# builf docker image, replace <docker_image_name> with your docker image name
docker build -t level_0 .
```
```bash
# deploy docker image in local server
docker Run --name  level_0 -p 8989:8989  level_0
```
### Test API
open your browser [predict penüing specie](http://localhost:8989/docs#/default/predict_penguins_predict_post)

-chose model
-if you want you can use body request [examples](http://localhost:8989/examples) to build body rrequest json
 
### Recomended tools
- Visual Studio Code

### Extensions for VScode
- Python
- Pylance
- flake8
- autoDocstring - python3.8 DocString
- python3.8 ident
- sonarlint
- git graph
- gitlens
- change-case
- json schema store catalog
- hasicorp terreform
- tabnine


