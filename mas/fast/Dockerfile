# Dockerfile
FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

# Copiar todos los archivos de tu aplicación en el directorio actual al /code en el contenedor
COPY . /code

# El comando para ejecutar la aplicación con Uvicorn. Ajusta la ruta del módulo si es necesario.
CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--reload"]