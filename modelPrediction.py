import os
import numpy as np
import joblib
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# === Paths ===
output_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\FinalMergedDataset"

# === Load dataset ===
df = pd.read_csv(os.path.join(output_path, "multi_hazard_cleaned.csv"))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Temporal split (70% train, 15% val, 15% test)
total_days = len(df)
train_end = int(total_days * 0.7)
val_end = int(total_days * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Train range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Validation range: {val_df['date'].min()} to {val_df['date'].max()}")
print(f"Test range: {test_df['date'].min()} to {test_df['date'].max()}")

# === Feature/Target split (adapt to your schema) ===
feature_cols = [c for c in df.columns if c not in [
    'date', 'flood', 'wildfire', 'heatwave', 'coldwave', 'drought'
]]

X_test = test_df[feature_cols]
y_test = {
    "flood": test_df['Flood_Count_anomaly'],
    "wildfire": test_df['Wildfire_Count_anomaly'],
    "heatwave": test_df['Heatwave_out_of_range'],
    "coldwave": test_df['Coldwave_out_of_range'],
    "drought": test_df['Drought_Index_anomaly'],
}

# === Load trained models ===
def load_models(hazard, horizon):
    """Load all trained models for a given hazard and horizon."""
    models = {}
    lgb_path = os.path.join(output_path, f"{hazard}_{horizon}_lgb.pkl")
    cat_path = os.path.join(output_path, f"{hazard}_{horizon}_cat.pkl")
    sarimax_path = os.path.join(output_path, f"{hazard}_{horizon}_sarimax.pkl")

    if os.path.exists(lgb_path):
        models["LightGBM"] = joblib.load(lgb_path)
    if os.path.exists(cat_path):
        models["CatBoost"] = joblib.load(cat_path)
    if hazard == "drought" and os.path.exists(sarimax_path):
        models["SARIMAX"] = SARIMAXResults.load(sarimax_path)

    print(f"Loaded models for {hazard}: {list(models.keys())}")
    return models

# === Run predictions ===
def run_predictions(hazard, horizon, X_test, y_true):
    """Run inference for a given hazard on test data."""
    models = load_models(hazard, horizon)

    if not models:
        print(f"No models found for {hazard}")
        return

    print(f"\n=== {hazard.upper()} ===")
    for name, model in models.items():
        if name in ["LightGBM", "CatBoost"]:
            preds = model.predict_proba(X_test)[:, 1]  # probability of disaster
            print(f"{name}_probabilities → {preds[:10]}")  # first 10 values
        elif name == "SARIMAX":
            # Clean exogenous test data
            exog_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
            forecast = model.forecast(steps=len(y_true), exog=exog_test)
            print(f"SARIMAX_forecast → {forecast[:10]}")

# === Hazards to evaluate ===
hazards = ["flood", "wildfire", "heatwave", "coldwave", "drought"]
horizon = "48h"  # adjust to match training

for hazard in hazards:
    run_predictions(hazard, horizon, X_test, y_test[hazard])
