# Multi-Disaster & Crime Forecasting System  
An AI-powered decision support system for **disaster risk reduction** and **crime prevention**. The project integrates **disaster forecasting, crime prediction, and dynamic resource allocation (DRA)** into a unified framework, enabling real-time insights, deployment planning, and post-event feedback.  

## Features  
- **Disaster Forecasting** – Predicts floods, droughts, heatwaves, wildfires, and coldwaves with event probability, severity, and confidence intervals.  
- **Crime Prediction** – Spatio-temporal crime risk forecasting with hotspot detection and patrol demand estimation.  
- **Dynamic Resource Allocation (DRA)** – Optimized distribution of emergency units, patrol schedules, and contingency planning.  
- **Modeling Approaches** – Includes advanced ML/DL models such as LightGBM, CatBoost, SARIMAX, LSTM, CNNs, U-Net, and transformers.  
- **Evaluation Metrics** – Accuracy, ROC-AUC, F1 Score, confusion matrices, and time-based cross-validation.  
- **Temporal Splits & Rolling CV** – Structured training/validation/testing pipelines with YAML + JSON export for reproducibility.  
- **Operational Outputs** – Deployment schedules, hotspot maps, patrol routes, and dashboards.  

## Tech Stack  
- **Languages & Frameworks:** Python, Pandas, Scikit-learn, PyTorch, TensorFlow, LightGBM, CatBoost  
- **Data Processing:** Pandas, NumPy, MODIS satellite data, crime datasets (NSL-KDD, UNSW-NB, CIC-IDS, NYC crime data)  
- **Model Types:** ML (LightGBM, CatBoost, SVM), DL (CNN, LSTM, Bi-LSTM, U-Net, VGG16, BERT), Time-Series (SARIMAX)  
- **Evaluation & Visualization:** Matplotlib, Seaborn, ROC-AUC, F1 Score, Confusion Matrix  
- **Deployment & Hosting:** GitHub (for research prototype), extensible to web dashboards/APIs  

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

## Development Notes  
- Implemented modular pipeline for disaster & crime forecasting  
- Integrated rolling CV for robust temporal generalization  
- Evaluated multiple ML/DL models with confusion matrices & ROC-AUC  
- Identified strengths (high accuracy on flood/wildfire/drought) and weaknesses (imbalanced data in heatwave/coldwave)  
- Prepared for extension into **real-time dashboards & decision-support tools**  

## Future Work  
- Expand dataset with additional disasters & crime sources  
- Integrate **LLM-based NER & NLP pipelines** for unstructured disaster reports  
- Build real-time **interactive dashboard** for operational use  
- Deploy APIs for government and emergency services  
