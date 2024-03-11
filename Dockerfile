# FROM apache/airflow:2.6.0
# USER airflow
# COPY requirements.txt /requirements.txt
# RUN pip install -r requirements.txt


# FROM apache/airflow:2.8.2
# COPY requirements.txt /
# RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt


FROM apache/airflow:2.8.2
COPY requirements.txt .
RUN pip install -r requirements.txt
