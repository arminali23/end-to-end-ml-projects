from fastapi import FastAPI
from app.schema import PassengerInput, PredictionOutput
from app import model

app = FastAPI(
    title="Titanic Survival Predictor",
    description="ML model served via FastAPI",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    result = model.predict(passenger.model_dump())
    return result