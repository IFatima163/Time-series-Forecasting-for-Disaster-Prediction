import pandas as pd
import numpy as np
import xarray as xr
import os
from scipy import stats

# -----------------------------
# Paths
# -----------------------------
individual_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\IndividualDisasterDatasets"
output_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\FinalMergedDataset"

# -----------------------------
# Load all hazard datasets
# -----------------------------
floods_daily = pd.read_csv(os.path.join(individual_path, "floods_daily.csv"))
drought_daily = pd.read_csv(os.path.join(individual_path, "drought_daily.csv"))
heatcold_daily = pd.read_csv(os.path.join(individual_path, "heatcold_daily.csv"))
fires_daily = pd.read_csv(os.path.join(individual_path, "fires_daily.csv"))

# Convert date columns to datetime.date
for df_ in [floods_daily, drought_daily, fires_daily, heatcold_daily]:
    df_["date"] = pd.to_datetime(df_["date"]).dt.date

# -----------------------------
# Define full date range
# -----------------------------
start_date = min(floods_daily["date"].min(),
                 drought_daily["date"].min(),
                 fires_daily["date"].min(),
                 heatcold_daily["date"].min())
end_date = max(floods_daily["date"].max(),
               drought_daily["date"].max(),
               fires_daily["date"].max(),
               heatcold_daily["date"].max())
full_dates = pd.date_range(start=start_date, end=end_date)
df = pd.DataFrame({"date": full_dates})
df["date"] = pd.to_datetime(df["date"]).dt.date

# -----------------------------
# Merge all hazard datasets
# -----------------------------
df = df.merge(floods_daily, on="date", how="left")
df = df.merge(drought_daily, on="date", how="left")
df = df.merge(heatcold_daily, on="date", how="left")
df = df.merge(fires_daily, on="date", how="left")

# -----------------------------
# Step 1: Hazard stats
# -----------------------------
stats = {}
for col in ["Flood_Count", "Wildfire_Count", "Heatwave", "Coldwave", "Drought_Index"]:
    if col in df.columns:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean()
        }
print("\nHazard statistics (min, max, mean):")
for h, s in stats.items():
    print(f"{h}: min={s['min']}, max={s['max']}, mean={s['mean']}")

# -----------------------------
# Step 2: Update heat/coldwave thresholds if temperature exists
# -----------------------------
if "tmax" in df.columns and "tmin" in df.columns:
    heat_thresh = df["tmax"].quantile(0.9)   # top 10% hottest days
    cold_thresh = df["tmin"].quantile(0.1)   # bottom 10% coldest days
    df["Heatwave"] = (df["tmax"] >= heat_thresh).astype(int)
    df["Coldwave"] = (df["tmin"] <= cold_thresh).astype(int)
else:
    df["Heatwave"] = df.get("Heatwave", 0).fillna(0).astype(int)
    df["Coldwave"] = df.get("Coldwave", 0).fillna(0).astype(int)

# -----------------------------
# Step 3: Fill remaining missing values
# -----------------------------
df["Flood_Count"] = df["Flood_Count"].fillna(0).astype(int)
df["Wildfire_Count"] = df["Wildfire_Count"].fillna(0).astype(int)
df["Drought_Index"] = df["Drought_Index"].ffill()

# -----------------------------
# Step 4: Quality Control & Cleaning
# -----------------------------

# Range Checks
limits = {
    "Flood_Count": (0, 5000),
    "Wildfire_Count": (0, 6000),
    "Drought_Index": (0, 100),
    "Heatwave": (0, 1),
    "Coldwave": (0, 1)
}
for col, (min_val, max_val) in limits.items():
    df[f"{col}_out_of_range"] = ((df[col] < min_val) | (df[col] > max_val)).astype(int)
    df[col] = df[col].clip(lower=min_val, upper=max_val)

# Duplicates
df["duplicate_row"] = df.duplicated(subset=["date"]).astype(int)
df = df.drop_duplicates(subset=["date"])

# Impossible jumps
max_daily_jump = {"Flood_Count": 1000, "Wildfire_Count": 2000, "Drought_Index": 20}
for col, jump in max_daily_jump.items():
    df[f"{col}_jump_flag"] = (df[col].diff().abs() > jump).astype(int)
    
# Harmonize Time & Units already done

# Missing Data Strategy
for col in ["Drought_Index"]:
    df[f"{col}_mask"] = df[col].isna().astype(int)
    df[col] = df[col].ffill()

# Outliers & Anomaly Flags using IQR
numeric_cols = ["Flood_Count", "Wildfire_Count", "Drought_Index"]
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[f"{col}_anomaly"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    df[col] = df[col].clip(lower=lower, upper=upper)

# -----------------------------
# Step 5: Sanity Checks
# -----------------------------
print("\nTotal Heatwave events:", df["Heatwave"].sum())
print("Total Coldwave events:", df["Coldwave"].sum())

# -----------------------------
# Step 6: Dataset Health Report
# -----------------------------
print("\nDataset Health Report")
print("-------------------------------")
print(f"Total rows: {len(df)}")
print(f"Total columns: {df.shape[1]}")
print("\nColumns and data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
hazards = ["Flood_Count", "Wildfire_Count", "Heatwave", "Coldwave", "Drought_Index"]
print("\nTotal events / sum per hazard:")
for h in hazards:
    if h in df.columns:
        print(f"{h}: {df[h].sum()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# -----------------------------
# Step 7: Summary of QC
# -----------------------------
qc_cols = [c for c in df.columns if "flag" in c or "anomaly" in c or "out_of_range" in c or "mask" in c]
qc_summary = df[qc_cols].sum()
print("\nQC / Anomaly Summary:")
print(qc_summary)

# -----------------------------
# Step 8: Export cleaned dataset
# -----------------------------
df.to_csv(os.path.join(output_path, "multi_hazard_cleaned.csv"), index=False)
print("\nCleaned multi-hazard dataset saved with QC flags.")


# ------------------------------------------------------
# Dataset Health Report
print("\nDataset Health Report")
print("-------------------------------")
print(f"Total rows: {len(df)}")
print(f"Total columns: {df.shape[1]}")

print("\nColumns and data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

hazards = ["Flood_Count", "Wildfire_Count", "Heatwave", "Coldwave", "Drought_Index"]
print("\nTotal events / sum per hazard:")
for h in hazards:
    if h in df.columns:
        print(f"{h}: {df[h].sum()}")

print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
