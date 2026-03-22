# EY Water Quality Forecasting Challenge 2026

**Platform:** Zindi  
**Yarışma:** EY AI & Data Challenge 2026 — Optimizing Clean Water Supply  
**Public LB Skoru:** 0.3029 mean R²  
**EY Baseline:** 0.20 mean R² → %50 iyileştirme  

## Problem
Güney Afrika nehirlerinden alınan ölçümlerle 3 hedef tahmini:
- Total Alkalinity (TA)
- Electrical Conductance (EC)  
- Dissolved Reactive Phosphorus (DRP)

## Yaklaşım

### Validasyon Stratejisi
- `GroupKFold(n=5, group=station_id)` — spatial generalization
- Station ID: koordinat bazlı MD5 hash (orijinal veri seti ID içermiyor)
- Temporal leakage bilinçli olarak kabul edildi (162 istasyonun 139'u
  tam 5 yıl veri içeriyor — temporal split data starvation yaratıyor)

### Feature Engineering
- Mevsimsellik: `month_sin`, `month_cos` (döngüsel kodlama)
- Uydu indeksleri: NDVI, NDWI, NDMI, MNDWI, NDTI, NDBI
- TerraClimate: `pet` (potansiyel evapotranspirasyon)

### Model Mimarisi
LightGBM + XGBoost + CatBoost üçlü ensemble  
OOF R²'ye göre ağırlıklı blend (kötü model otomatik ezilir)  
DRP için Tweedie loss (sağa çarpık, sıfır-yoğun dağılım)

### Sonuçlar

| Model | TA | EC | DRP | Ortalama |
|-------|----|----|-----|----------|
| LightGBM | +0.3161 | +0.4059 | +0.1464 | — |
| XGBoost  | +0.2791 | +0.3861 | +0.1764 | — |
| CatBoost | +0.3908 | +0.3941 | +0.2228 | — |
| **Ensemble** | **+0.3630** | **+0.4198** | **+0.2032** | **+0.3286** |

## Mimari Notlar
- `data_loader.py`: Test setinden sıfır satır düşürme garantisi
- `validation.py`: Spatial leakage testi + fold dondurma
- `train.py`: OOF-ağırlıklı blend, Zindi format otomatik tespiti
- Çok-ajanlı War Room: Claude (CTO) + ChatGPT + DeepSeek + Gemini

## Çalıştırma
pip install -r requirements.txt
python -m src.train --data_dir data/raw --output_dir outputs/
