# EY Water Quality Forecasting Challenge 2026

**Platform:** Zindi  
**Competition:** EY AI & Data Challenge 2026 — Optimizing Clean Water Supply  
**Public LB Score:** 0.3029 mean R²  
**EY Baseline:** 0.20 mean R² → 50% improvement over baseline  

## Problem
Predict 3 water quality targets from measurements taken at South African rivers:
- Total Alkalinity (TA)
- Electrical Conductance (EC)
- Dissolved Reactive Phosphorus (DRP)

## Approach

### Validation Strategy
- `GroupKFold(n=5, group=station_id)` — spatial generalization
- Station ID derived via MD5 hash of coordinates (original dataset has no ID column)
- Temporal leakage intentionally accepted: 139 of 162 stations contain full
  5-year records — temporal splitting causes data starvation without meaningful gain

### Feature Engineering
- Seasonality: `month_sin`, `month_cos` (cyclic encoding)
- Satellite indices: NDVI, NDWI, NDMI, MNDWI, NDTI, NDBI
- TerraClimate: `pet` (potential evapotranspiration)

### Model Architecture
Triple ensemble: LightGBM + XGBoost + CatBoost  
OOF R²-weighted blend — underperforming models are automatically down-weighted  
Tweedie loss for DRP target (right-skewed, zero-inflated distribution)

### Results

| Model | TA | EC | DRP | Mean |
|-------|----|----|-----|------|
| LightGBM | +0.3161 | +0.4059 | +0.1464 | — |
| XGBoost  | +0.2791 | +0.3861 | +0.1764 | — |
| CatBoost | +0.3908 | +0.3941 | +0.2228 | — |
| **Ensemble** | **+0.3630** | **+0.4198** | **+0.2032** | **+0.3286** |

## Architecture Notes
- `data_loader.py`: Zero row-drop guarantee on test set
- `validation.py`: Spatial leakage test + fold freezing for reproducibility
- `train.py`: OOF-weighted blend, automatic Zindi format detection
- Multi-agent War Room workflow: Claude (CTO) + ChatGPT + DeepSeek + Gemini

## How to Run
```bash
pip install -r requirements.txt
python -m src.train --data_dir data/raw --output_dir outputs/
```

> **Note:** Raw data is not included in this repository due to Zindi's
> competition data policy. Download the dataset directly from the
> [competition page](https://zindi.africa/competitions/ey-water-quality-forecasting-challenge).
