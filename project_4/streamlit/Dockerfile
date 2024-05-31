FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./

RUN pip install -r requirements.txt

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]