
Modificaciones en el docker compose:

    - linea 53 -> apagar la imagen y crear un docker file para que se instale el requeriments 
    - linea 79 -> se agrega una linea de codigo para guardar los datos de los pinguinos
        - ${AIRFLOW_PROJ_DIR:-.}/datos:/opt/airflow/datos 
    - linea 298 -> agrego la base mysql para cumplir el punto 1 (Cree una instancia de una base de datos de preferencia (sugerencia: mysql))
    - linea 319 -> se agrega el volumen para la base mysql

Inspirado en: 

https://github.com/QPC-github/public-datasets-pipelines/blob/93063b9b104d2df3a11b8ad356559cb22ebaec15/datasets/ml_datasets/pipelines/penguins/penguins_dag.py

https://github.com/DanilBaibak/ml-in-production/tree/master

https://github.com/CharlieSergeant/airflow-minio-postgres-fastapi/tree/main

https://www.youtube.com/watch?v=lWkJ-03k6NM

https://www.youtube.com/watch?v=2tOhTGBWZXY

https://www.red-gate.com/simple-talk/databases/mysql/retrieving-mysql-data-python/

https://www.atlassian.com/data/notebook/how-to-execute-raw-sql-in-sqlalchemy



DAGS necesarios:

Cargar datos de penguins, sin preprocesamiento!   (ok)
Borrar contenido base de datos (ok)
Realizar entrenamiento de modelo usando datos de la base de datos (realizando procesamiento) 


docker-compose up airflow-init
docker-compose up