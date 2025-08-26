import os
import joblib
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.dummy import DummyClassifier

# === Paths ===
output_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\FinalMergedDataset"
os.makedirs(output_path, exist_ok=True)

def train_models(hazard, horizon, X_train, y_train, exog=None):
    """
    Train models per hazard & horizon.
    - Tree models (LightGBM, CatBoost) → all hazards
    - Classical (SARIMAX) → long-horizon drought
    - DummyClassifier if only one class in target
    Saves models to disk and returns dict of trained models.
    """
    models = {}

    # === Handle trivial case: only one class in target ===
    if y_train.nunique() < 2:
        print(f"⚠️ {hazard} has only one class → training DummyClassifier instead.")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        models["Dummy"] = dummy
        joblib.dump(dummy, os.path.join(output_path, f"{hazard}_{horizon}_dummy.pkl"))
        return models

    # --- Tree Model: LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=-1
    )
    lgb_model.fit(X_train, y_train)
    models["LightGBM"] = lgb_model
    joblib.dump(lgb_model, os.path.join(output_path, f"{hazard}_{horizon}_lgb.pkl"))

    # --- Tree Model: CatBoost ---
    cat_model = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models["CatBoost"] = cat_model
    joblib.dump(cat_model, os.path.join(output_path, f"{hazard}_{horizon}_cat.pkl"))

    # --- Classical Model: SARIMAX ---
    if hazard == "drought":
        # Clean exogenous data (drop or fill NaN/Inf)
        if exog is not None:
            exog = exog.replace([np.inf, -np.inf], np.nan).fillna(0)  # safe fill strategy
        
        y_train_clean = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)

        # SARIMAX works on time-series y + exogenous regressors
        sarimax = SARIMAX(y_train_clean, exog=exog, order=(2,1,2), seasonal_order=(1,0,1,12))
        sarimax_fit = sarimax.fit(disp=False)
        models["SARIMAX"] = sarimax_fit
        sarimax_fit.save(os.path.join(output_path, f"{hazard}_{horizon}_sarimax.pkl"))



# === Example Call ===
if __name__ == "__main__":
    import pandas as pd

    # === Load dataset ===
    df = pd.read_csv(os.path.join(output_path, "multi_hazard_cleaned.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Temporal split
    total_days = len(df)
    train_end = int(total_days * 0.7)
    val_end = int(total_days * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save splits for reference
    train_df.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_path, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_data.csv"), index=False)

    # === Feature/Target split (adapt to your columns) ===
    feature_cols = [c for c in df.columns if c not in ['date', 'flood', 'wildfire', 'heatwave', 'coldwave', 'drought']]
    
    hazards = {
        "flood": "Flood_Count_anomaly",
        "wildfire": "Wildfire_Count_anomaly",
        "heatwave": "Heatwave_out_of_range",
        "coldwave": "Coldwave_out_of_range",
        "drought": "Drought_Index_anomaly"
    }

    horizon = "48h"

    # Train per hazard
    for hazard, target_col in hazards.items():
        print(f"\nTraining models for {hazard}...")
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        exog = None
        if hazard == "drought":  # SARIMAX needs exogenous features
            exog = X_train

        train_models(hazard, horizon, X_train, y_train, exog)
        print(f"✅ {hazard} models saved.")
