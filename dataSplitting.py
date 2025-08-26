import pandas as pd
import json
import yaml
import os


# Load the cleaned dataset
output_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\FinalMergedDataset"
df = pd.read_csv(os.path.join(output_path, "multi_hazard_cleaned.csv"))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Define train/validation/test split by time
total_days = len(df)
train_end = int(total_days * 0.7)
val_end = int(total_days * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Train range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Validation range: {val_df['date'].min()} to {val_df['date'].max()}")
print(f"Test range: {test_df['date'].min()} to {test_df['date'].max()}")

# Define rolling cross-validation folds
rolling_folds = []
history_length = pd.Timedelta(days=365*5)  # 5 years for initial train
test_length = pd.Timedelta(days=365)       # 1 year for test

start_date = df['date'].min() + history_length
end_date = df['date'].max() - test_length

current_start = start_date
while current_start <= end_date:
    train_fold = df[df['date'] < current_start]
    test_fold = df[(df['date'] >= current_start) & (df['date'] < current_start + test_length)]
    rolling_folds.append({
        'train_start': str(train_fold['date'].min().date()),
        'train_end': str(train_fold['date'].max().date()),
        'test_start': str(test_fold['date'].min().date()),
        'test_end': str(test_fold['date'].max().date())
    })
    current_start += test_length  # roll forward 1 year

# Export temporal split and CV fold definitions

# Temporal split YAML
temporal_split = {
    'train': {'start': str(train_df['date'].min().date()), 'end': str(train_df['date'].max().date())},
    'validation': {'start': str(val_df['date'].min().date()), 'end': str(val_df['date'].max().date())},
    'test': {'start': str(test_df['date'].min().date()), 'end': str(test_df['date'].max().date())}
}

with open(os.path.join(output_path, 'splits/temporal.yaml'), 'w') as f:
    yaml.dump(temporal_split, f)

# Rolling CV folds JSON
with open(os.path.join(output_path, 'splits/cv_folds.json'), 'w') as f:
    json.dump(rolling_folds, f, indent=4)

print("Temporal split and rolling CV folds saved successfully!")
