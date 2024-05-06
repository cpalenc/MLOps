from fastapi import FastAPI

# Crea una instancia de la aplicación FastAPI
app = FastAPI()

# Define una ruta para un endpoint
@app.get("/hello")
def hello_world(name: str):
    return {"message": f"Hello, {name}!"}
