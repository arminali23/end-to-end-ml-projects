import joblib
import json
import pandas as pd
import numpy as np

pipeline = joblib.load("artifacts/model.pkl")

with open("artifacts/report.json", "r") as f:
    report = json.load(f)
    
BASE_FEATURES = [
    "crim", "zn", "indus", "chas", "nox", "rm",
    "age", "dis", "rad", "tax", "ptratio", "b", "lstat"
]

def predict(data: dict) -> dict:
    df = pd.DataFrame([data])[BASE_FEATURES]

    df["room_density"]   = df["rm"] / df["lstat"]
    df["tax_rate_ratio"] = df["tax"] / df["ptratio"]
    df["age_crime"]      = df["age"] * df["crim"]

    price_k   = float(pipeline.predict(df)[0])
    price_usd = round(price_k * 1000, 2)

    return {
        "predicted_price_usd": price_usd,
        "predicted_price_k":   round(price_k, 4),
        "model_used":          report["best_model"],
        "r2_score":            report["final_train_r2"],
    }