from fastapi import FastAPI

app = FastAPI(title="Bug Exorcist Backend")

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "service": "Bug Exorcist"
    }
