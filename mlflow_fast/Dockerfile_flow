# Use an official MLflow runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install MLflow and other dependencies
RUN pip install mlflow

# Expose the MLflow server port
EXPOSE 5000

# Run the MLflow server when the container launches
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]