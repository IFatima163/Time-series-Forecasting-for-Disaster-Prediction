# Multi-Disaster Forecasting System  
An AI-powered decision support system for **disaster risk reduction**. The project enables real-time insights for disaster occurences.  

## Features  
- **Disaster Forecasting** – Predicts floods, droughts, heatwaves, wildfires, and coldwaves with event probability, severity, and confidence intervals.    
- **Evaluation Metrics** – Accuracy, ROC-AUC, F1 Score, confusion matrices, and time-based cross-validation.  
- **Temporal Splits & Rolling CV** – Structured training/validation/testing pipelines with YAML + JSON export for reproducibility.  

## Tech Stack  
- **Languages & Frameworks:** Python, Pandas, Scikit-learn, PyTorch, TensorFlow, LightGBM, CatBoost  
- **Data Processing:** Pandas, NumPy, MODIS satellite data, crime datasets (NSL-KDD, UNSW-NB, CIC-IDS, NYC crime data)  
- **Model Types:** ML (LightGBM, CatBoost, SVM), Time-Series (SARIMAX)  
- **Evaluation & Visualization:** Matplotlib, Seaborn, ROC-AUC, F1 Score, Confusion Matrix  

## Dataset & Preprocessing  
- Multi-disaster dataset merged, cleaned, and temporally sorted (`multi_disaster_cleaned.csv`)  
- Automated **train/validation/test splits** (70/15/15)  
- Rolling cross-validation with **5-year history & 1-year test windows**  
- Splits exported to:  
  - `splits/temporal.yaml`  
  - `splits/cv_folds.json`  

## Results  

### Disaster Forecasting  
| Disaster   | Model(s)              | Accuracy | ROC-AUC | Notes |  
|------------|-----------------------|----------|---------|-------|  
| Flood      | LightGBM, CatBoost    | **1.0**  | **1.0** | Perfect classification (no false positives/negatives). |  
| Wildfire   | LightGBM, CatBoost    | **1.0**  | **1.0** | Perfect classification, both models identical. |  
| Heatwave   | LightGBM, Dummy       | **1.0**  | NaN     | Only one class present → ROC-AUC undefined. Data imbalance issue. |  
| Coldwave   | Dummy                 | **1.0**  | NaN     | Same issue: single-class dataset, trivial accuracy. |  
| Drought    | LightGBM, CatBoost, SARIMAX | **1.0** | 1.0 (Cat ~0.9999) | Strong results; SARIMAX unsuitable for classification metrics. |  

**Strengths**: Flood, Wildfire, and Drought forecasting show robust performance with zero errors.  
**Weaknesses**: Heatwave & Coldwave datasets lack event diversity → ROC-AUC not meaningful, risk of overfitting.  
