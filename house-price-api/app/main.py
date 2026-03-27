import json
from fastapi import FastAPI
from app.schema import HouseInput, PredictionOutput
from app import model

with open("artifacts/report.json") as f:
    report = json.load(f)

app = FastAPI(
    title="House Price Predictor",
    description="Regression model with automatic model selection",
    version="2.0.0"
)

@app.get("/healt")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return report

@app.post("/predict", response_model=PredictionOutput)
def predict(house: HouseInput):
    result = model.predict(house.model_dump())
    return result