from fastapi import FastAPI
from .database import engine
from . import models

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bug Exorcist Backend is running"}
