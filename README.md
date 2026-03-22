# 🌍 Geospatial Water Quality Predictor (EY Challenge)

## 🧠 Architecture Overview
A predictive modeling framework developed to forecast complex chemical concentrations (e.g., DRP) in water ecosystems. The system fuses physical sensor data with remote sensing satellite indices to map non-linear environmental impacts.

## 🛠️ Tech Stack
* **Core Models:** XGBoost & LightGBM Ensembles
* **Hyperparameter Tuning:** Optuna
* **Domain:** Environmental Data Science, IoT Sensor Analysis

## 🎯 Core Mechanisms
* **Geospatial Feature Engineering:** Integration of satellite indices including NDVI, NDWI, EVI, and BSI.
* **Temporal Dynamics:** Engineered complex 'Lag Features' and shift operations to capture delayed environmental impacts (e.g., past rainfall affecting current water quality).
* **Advanced Imputation:** Handled missing sensor data and anomalies through dynamic imputation strategies before feeding them into the boosting algorithms.
