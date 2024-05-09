# Proyecto 3 - Grupo 1

![alt text](images/logo.PNG)

Class: MLOps <br>
Code: 11179 <br>
Professor: Cristian Diaz Alvarez <br>
Members:

    Daniel Chavarro - @anielFchavarro
    Cristhian Palencia - @cpalenc
    Oscar Correa - @oecorrechag

## Enlace a video
[Explicacion de proyecto 3!](https://youtu.be/EkEZTNt3wLk)

## Tabla de contenido

- <a href='#1'>Resultados del Proyecto 3 </a>
- <a href='#2'>Locust with Docker compose </a>
- <a href='#3'>Servicios </a>
- <a href='#3.1'>Airflow </a>
- <a href='#3.2'>MLflow </a>
- <a href='#4'>Ingeniería de Datos </a>
- <a href='#5'>Entrenamiento </a>
- <a href='#6'>Despliegue de los Servicios </a>

### <a id='1'>Resultados del Proyecto 3 </a>

En este documento se presenta el desarrollo del Proyecto 3, el cual se enfocó en la creación de un sistema de MLOps para realizar inferencias sobre datos médicos con el propósito de determinar si una persona será readmitida al hospital dentro de los 30 días siguientes a su última visita.

Para proporcionar un contexto más detallado, se ha creado un vídeo que explora en profundidad el desarrollo de este proyecto.

### <a id='2'>Arquitectura </a>

Para este caso, se optó por construir una red de servicios que integrara varios componentes mediante un archivo de Docker Compose. Este archivo se encargaría de desplegar todos los servicios diseñados. Además, se tenía la intención de utilizar este archivo como base para la posterior creación de los archivos necesarios utilizando el paquete Kompose. Sin embargo, a pesar de que se generaron con éxito los archivos para Kubernetes, estos no pudieron ser utilizados debido a problemas de memoria. Kubernetes generaba imágenes y levantaba contenedores de manera constante, agotando los recursos de la máquina y evitando que los servicios pudieran ser desplegados utilizando los archivos de Kubernetes.

Por otro lado, es importante mencionar que se lograron implementar las asignaciones restantes, que incluyen el uso de Airflow como orquestador, MLflow, preprocesamiento de datos, entrenamiento de datos, creación de una API de inferencia e interfaz para la inferencia. A continuación, se detallan los servicios desplegados.

### <a id='3'>Servicios </a>

#### <a id='3.1'>Airflow </a>

Para este servicio, se desarrollaron tres DAGs: uno para la ingesta de datos, otro para el procesamiento de datos y el último para el entrenamiento de modelos. Además, el DAG de entrenamiento permitió registrar el modelo seleccionado en la aplicación de MLflow.

#### <a id='3.2'>MLflow </a>

Este servicio se utilizó para registrar los experimentos resultantes de los entrenamientos.

### <a id='4'>Ingeniería de Datos </a>

Para esta fase del proyecto, se optó por crear dos bases de datos: Raw_Data y Clean_data, en las cuales se almacenó la información recibida y transformada respectivamente. Además, se realizó un análisis exploratorio de los datos, el cual proporcionó insights para la posterior limpieza y transformación de los datos. Todo este proceso se ejecutó con la ayuda de los DAGs previamente mencionados.

### <a id='5'>Entrenamiento </a>

En esta etapa, se entrenaron varios modelos de clasificación, de los cuales se seleccionó aquel con el mejor rendimiento en términos de precisión (accuracy). Este modelo se almacenó en MLflow para el seguimiento de experimentos.

### <a id='6'>Despliegue de los Servicios </a>

Como se mencionó anteriormente, no fue posible desplegar los servicios utilizando Kubernetes. En su lugar, se utilizó un archivo de Docker Compose que debe ser levantado de la siguiente manera:

```bash
docker-compose up -d
```

<hr>

[Go to Top](#Table-of-Contents)
