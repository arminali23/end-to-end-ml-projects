import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Load once when the app starts
pipeline = joblib.load("artifacts/model.pkl")

def predict(data: dict) -> dict:
    df = pd.DataFrame([data])[FEATURES]
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 4)
    }