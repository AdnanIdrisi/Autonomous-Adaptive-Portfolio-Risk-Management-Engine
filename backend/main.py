from fastapi import FastAPI
from backend.api.routes import router

app = FastAPI(
    title="Adaptive Portfolio & Risk Management Engine",
    description="Autonomous ML-driven portfolio allocation and risk control system",
    version="1.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Adaptive Portfolio Engine API is running 🚀"
    }
