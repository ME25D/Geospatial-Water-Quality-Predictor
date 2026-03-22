## Financial Health Prediction Challenge
**Platform:** Zindi | **Skor:** 0.8920 F1 (Public LB)

### Yaklaşım
- LightGBM + XGBoost + CatBoost ensemble
- OOF-ağırlıklı blend
- Stratified K-Fold CV

### Sonuçlar
| Model | CV F1 |
|-------|-------|
| LightGBM | x.xxx |
| XGBoost  | x.xxx |
| Ensemble | 0.892 |

### Çalıştırma
pip install -r requirements.txt
python src/train.py --data_dir data/raw
```

---

### `.gitignore` dosyası oluştur

Repo'nun köküne bunu ekle:
```
data/
*.csv
*.pkl
*.joblib
__pycache__/
.ipynb_checkpoints/
outputs/submission*
```

---

### requirements.txt
```
lightgbm
xgboost
catboost
scikit-learn
pandas
numpy
