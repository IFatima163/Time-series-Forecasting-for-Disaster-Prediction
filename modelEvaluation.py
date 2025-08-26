import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# === Paths ===
output_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\FinalMergedDataset"
test_data_path = os.path.join(output_path, "test_data.csv")

# === Load test data ===
test_df = pd.read_csv(test_data_path, parse_dates=["date"], index_col="date")
feature_cols = [c for c in test_df.columns if c not in ['flood', 'wildfire', 'heatwave', 'coldwave', 'drought']]

X_test = test_df[feature_cols]
y_test = {
    "flood": test_df['Flood_Count_anomaly'],
    "wildfire": test_df['Wildfire_Count_anomaly'],
    "heatwave": test_df['Heatwave_out_of_range'],
    "coldwave": test_df['Coldwave_out_of_range'],
    "drought": test_df['Drought_Index_anomaly'],
}

horizon = "48h"

# === Load models ===
def load_models(hazard, horizon):
    models = {}
    for model_name in ["lgb", "cat", "sarimax", "dummy"]:
        path = os.path.join(output_path, f"{hazard}_{horizon}_{model_name}.pkl")
        if os.path.exists(path):
            if model_name == "sarimax":
                from statsmodels.tsa.statespace.sarimax import SARIMAXResults
                models["SARIMAX"] = SARIMAXResults.load(path)
            else:
                models[model_name.capitalize() if model_name!="lgb" else "LightGBM"] = joblib.load(path)
    print(f"Loaded models for {hazard}: {list(models.keys())}")
    return models

# === Evaluate & Plot ===
def predict_and_evaluate(hazard, X_test, y_true):
    models = load_models(hazard, horizon)
    if not models:
        print(f"No models found for {hazard}")
        return

    print(f"\n=== {hazard.upper()} ===")
    plt.figure(figsize=(12,5))
    
    # Add tiny jitter if y_true is constant to make line visible
    y_true_plot = y_true.copy()
    if y_true.nunique() == 1:
        y_true_plot = y_true + np.random.normal(0, 1e-6, size=len(y_true))
    
    plt.plot(y_true_plot.values, label="Actual", color="blue", linewidth=2)

    # Inside predict_and_evaluate function
    for name, model in models.items():
        if name in ["LightGBM", "Cat", "Dummy"]:
            # Handle single-class safely
            if hasattr(model, "predict_proba"):
                if hasattr(model, "classes_") and model.classes_.size == 1:
                    y_pred_proba = np.full(len(X_test), fill_value=model.classes_[0])
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict(X_test)

            # Accuracy & ROC-AUC
            try:
                acc = accuracy_score(y_true, np.round(y_pred_proba))
                roc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else np.nan
            except Exception:
                acc, roc = np.nan, np.nan

        elif name == "SARIMAX":
            # SARIMAX requires start/end, exog only if trained with it
            if hasattr(model.model, "exog") and model.model.exog is not None:
                exog = X_test.values  # same columns as training
            else:
                exog = None

            # Forecast over entire test period
            y_pred_proba = model.get_forecast(steps=len(y_true), exog=exog).predicted_mean

            # For SARIMAX, compute MSE instead of accuracy
            acc = roc = np.nan

        print(f"{name} Accuracy: {acc}")
        print(f"{name} ROC-AUC: {roc}")

        print(f"{name} Confusion Matrix:\n{confusion_matrix(y_true, np.round(y_pred_proba))}\n")
        
        plt.plot(y_pred_proba, label=f"{name} Predicted", alpha=0.7)

    plt.title(f"{hazard.upper()} Predictions vs Actual")
    plt.xlabel("Time Index")
    plt.ylabel("Probability / Value")
    plt.legend()
    plt.show()


# === Run evaluation for all hazards ===
hazards = ["flood", "wildfire", "heatwave", "coldwave", "drought"]
for hazard in hazards:
    predict_and_evaluate(hazard, X_test, y_test[hazard])
