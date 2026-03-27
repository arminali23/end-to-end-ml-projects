import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

df["room_density"]   = df["rm"] / df["lstat"]       
df["tax_rate_ratio"] = df["tax"] / df["ptratio"]   
df["age_crime"]      = df["age"] * df["crim"]

TARGET = "medv"

FEATURES = [
    "crim", "zn", "indus", "chas", "nox", "rm",
    "age", "dis", "rad", "tax", "ptratio", "b", "lstat",
    "room_density", "tax_rate_ratio", "age_crime"
]

X = df[FEATURES]
y = df[TARGET]

preprocessor = ColumnTransformer(transformers = [
  ("num", Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ]),FEATURES)  
])

candidates = {
    "ridge": Ridge(alpha=1.0),
    "rf": RandomForestRegressor(n_estimators=100, random_state=42),
    "gbm": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("model comparison")
results = {}

for name, regressor in candidates.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv =5, 
        scoring="neg_root_mean_squared_error"
    )
    
    rmse_scores = -cv_scores
    results[name] = {
        "mean_rmse": round(float(rmse_scores.mean()), 4),
        "std_rmse":  round(float(rmse_scores.std()),  4),
    }
    
    print(f"{name:>6}  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

best_name = min(results, key=lambda k: results[k]["mean_rmse"])
print(f"best model:", {best_name})

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", candidates[best_name])
])

best_pipeline.fit(X,y)

y_pred = best_pipeline.predict(X)
final_rmse = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)
final_r2   = round(float(r2_score(y, y_pred)), 4)

print(f"   Train RMSE : {final_rmse}")
print(f"   Train R²   : {final_r2}")

os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_pipeline, "artifacts/model.pkl")

report = {
    "best_model":      best_name,
    "features":        FEATURES,
    "cv_comparison":   results,
    "final_train_rmse": final_rmse,
    "final_train_r2":   final_r2,
}

with open("artifacts/report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n Model saved to artifacts/model.pkl")
print("Report saved to artifacts/report.json")
