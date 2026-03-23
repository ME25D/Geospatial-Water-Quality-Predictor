"""
src/train.py  —  LightGBM + XGBoost + CatBoost  |  OOF-Ağırlıklı Ensemble

Submission kısmı "kurşun geçirmez" formata alındı:
  - Zindi'nin gerçek SampleSubmission.csv'si otomatik okunur
  - Format otomatik tespit edilir (3 senaryo):
      A) Wide  : ID sütunu + TA/EC/DRP sütunları
      B) Long  : ID sütunu + tek Value sütunu (melted)
      C) Coord : Latitude/Longitude/Sample Date (yerel şablon)
  - Eksik ID'ler train ortalamasıyla doldurulur (Zindi boş ID reddeder)

Çalıştır:
  python -m src.train --data_dir data/raw --output_dir outputs/
  python -m src.train --data_dir data/raw --sample_sub data/raw/SampleSubmission.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ── Kütüphane import'ları ─────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logging.warning("LightGBM yok → pip install lightgbm")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logging.warning("XGBoost yok → pip install xgboost")

try:
    import catboost as cb
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    logging.warning("CatBoost yok → pip install catboost")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import FEATURE_COLS, TARGET_COLS, WaterQualityDataLoader
from src.validation  import WaterQualityGroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────
_SEED            = 42
_CV_N_SPLITS     = 5
_ES              = 50
_SEASON_COLS     = ["month_sin", "month_cos"]
ALL_FEATURE_COLS = FEATURE_COLS + _SEASON_COLS
_SHORT = {
    "Total Alkalinity":              "TA",
    "Electrical Conductance":        "EC",
    "Dissolved Reactive Phosphorus": "DRP",
}
# Zindi'nin sütun adı → bizim TARGET_COLS eşlemesi (küçük harf, boşluksuz karşılaştırma)
_TARGET_ALIASES = {
    "total alkalinity":              "Total Alkalinity",
    "total_alkalinity":              "Total Alkalinity",
    "totalalkalinity":               "Total Alkalinity",
    "ta":                            "Total Alkalinity",
    "electrical conductance":        "Electrical Conductance",
    "electrical_conductance":        "Electrical Conductance",
    "electricalconductance":         "Electrical Conductance",
    "ec":                            "Electrical Conductance",
    "dissolved reactive phosphorus": "Dissolved Reactive Phosphorus",
    "dissolved_reactive_phosphorus": "Dissolved Reactive Phosphorus",
    "dissolvedreactivephosphorus":   "Dissolved Reactive Phosphorus",
    "drp":                           "Dissolved Reactive Phosphorus",
}

# Olası Zindi Value sütun adları (melted format için)
_VALUE_COL_CANDIDATES = ["Value", "value", "target", "Target", "prediction", "Prediction"]


# ── Mevsimsellik ──────────────────────────────────────────────────────────────

def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce").dt.month
    out   = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


# ── Model parametreleri ───────────────────────────────────────────────────────

def _lgb_params(target: str) -> dict:
    base = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=-1, random_state=_SEED, verbose=-1)
    if "Phosphorus" in target:
        return {**base, "objective": "tweedie", "metric": "rmse",
                "tweedie_variance_power": 1.5}
    return {**base, "objective": "regression", "metric": "rmse"}


def _xgb_params(target: str) -> dict:
    base = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, random_state=_SEED, verbosity=0,
                early_stopping_rounds=_ES)
    if "Phosphorus" in target:
        return {**base, "objective": "reg:tweedie", "eval_metric": "rmse",
                "tweedie_variance_power": 1.5}
    return {**base, "objective": "reg:squarederror", "eval_metric": "rmse"}


def _cat_params(target: str) -> dict:
    base = dict(iterations=1000, learning_rate=0.05, depth=6,
                subsample=0.8, l2_leaf_reg=3.0, random_seed=_SEED,
                verbose=False, early_stopping_rounds=_ES)
    if "Phosphorus" in target:
        return {**base, "loss_function": "Tweedie:variance_power=1.5",
                "eval_metric": "RMSE"}
    return {**base, "loss_function": "RMSE", "eval_metric": "RMSE"}


# ── Fold fit+predict ──────────────────────────────────────────────────────────

def _run_lgb(Xtr, ytr, Xval, yval, Xtest, target):
    m = lgb.LGBMRegressor(**_lgb_params(target))
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(_ES, verbose=False),
                     lgb.log_evaluation(period=-1)])
    return m.predict(Xval), m.predict(Xtest), m.best_iteration_


def _run_xgb(Xtr, ytr, Xval, yval, Xtest, target):
    m = xgb.XGBRegressor(**_xgb_params(target))
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return m.predict(Xval), m.predict(Xtest), getattr(m, "best_iteration", -1)


def _run_cat(Xtr, ytr, Xval, yval, Xtest, target):
    m = cb.CatBoostRegressor(**_cat_params(target))
    m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True, verbose=False)
    return m.predict(Xval), m.predict(Xtest), m.best_iteration_


_MODELS: List[Tuple[str, bool, Callable]] = [
    ("LightGBM", _HAS_LGB, _run_lgb),
    ("XGBoost",  _HAS_XGB, _run_xgb),
    ("CatBoost", _HAS_CAT, _run_cat),
]


# ── OOF-Ağırlıklı Blend ──────────────────────────────────────────────────────

def _weighted_blend(
    stores:     Dict[str, Dict[str, np.ndarray]],
    active:     List[Tuple[str, Callable]],
    oof_r2_map: Dict[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """OOF R²'ye göre ağırlıklı ensemble. Negatif R² → 0 ağırlık."""
    blended: Dict[str, np.ndarray] = {}
    for target in TARGET_COLS:
        tag = _SHORT[target]
        raw_w = np.array([max(oof_r2_map[n][tag], 0.0) for n, _ in active])
        if raw_w.sum() == 0:
            logger.warning("%s: tüm OOF R² ≤ 0 → eşit ağırlık.", tag)
            raw_w = np.ones(len(active))
        w = raw_w / raw_w.sum()
        w_str = " | ".join(f"{n}={v:.3f}" for (n, _), v in zip(active, w))
        logger.info("  [BLEND] %s: %s", tag, w_str)
        stacked  = np.stack([stores[n][target] for n, _ in active], axis=1)
        blended[target] = stacked @ w
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# KURŞUN GEÇİRMEZ SUBMISSION  —  3 FORMAT SENARYOSU
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(sample_sub: pd.DataFrame) -> str:
    """
    Zindi submission şablonunun formatını otomatik tespit eder.

    Döndürür
    --------
    "coord"  → Latitude/Longitude/Sample Date sütunları var (yerel şablon)
    "wide"   → ID sütunu + hedef sütunları var (geniş format)
    "long"   → ID sütunu + tek Value sütunu (eriyik/melted format)
    """
    cols_lower = [c.lower() for c in sample_sub.columns]

    # Koordinat bazlı?
    if "latitude" in cols_lower and "longitude" in cols_lower:
        return "coord"

    # ID sütunu var mı?
    id_col = next((c for c in sample_sub.columns
                   if c.lower() in ("id", "row_id", "rowid", "sample_id")), None)
    if id_col is None:
        # İlk sütun ID gibi görünüyor mu?
        first = sample_sub.columns[0]
        if sample_sub[first].dtype == object:
            id_col = first

    if id_col is None:
        logger.warning("ID sütunu tespit edilemedi — coord formatı varsayılıyor.")
        return "coord"

    # Long mu wide mi?
    value_col = next((c for c in sample_sub.columns if c in _VALUE_COL_CANDIDATES), None)
    n_target_cols = sum(
        1 for c in sample_sub.columns
        if c.lower().replace(" ", "").replace("_", "") in
           [k.replace(" ", "").replace("_", "") for k in _TARGET_ALIASES]
    )

    if value_col and n_target_cols < 2:
        return "long"
    return "wide"


def _normalize_target_col(col_name: str) -> Optional[str]:
    """Sütun adını normalize edip TARGET_COLS'daki karşılığını döndürür."""
    key = col_name.lower().replace(" ", "").replace("_", "")
    for alias_key, canonical in _TARGET_ALIASES.items():
        if key == alias_key.replace(" ", "").replace("_", ""):
            return canonical
    return None


def _build_submission(
    sample_sub:   pd.DataFrame,
    pred_df:      pd.DataFrame,   # 200 satır: Latitude, Longitude, Sample Date + TARGET_COLS
    train_means:  Dict[str, float],
    fmt:          str,
) -> pd.DataFrame:
    """
    Tahmin DataFrame'ini Zindi şablonuna hizala.

    Parameters
    ----------
    sample_sub  : Zindi'nin orijinal şablon DataFrame'i
    pred_df     : bizim 200 satırlık tahmin çıktımız
    train_means : fallback NaN doldurucu
    fmt         : "coord" | "wide" | "long"
    """
    sub = sample_sub.copy()

    # ── Senaryo C: Koordinat bazlı (yerel şablon) ────────────────────────────
    if fmt == "coord":
        logger.info("Submission format: COORD (Latitude/Longitude/Sample Date)")
        join_key = ["Latitude", "Longitude", "Sample Date"]

        # Left merge: şablon satırları korunur, tahminler eklenir
        sub = sub.drop(columns=[c for c in TARGET_COLS if c in sub.columns])
        sub = sub.merge(pred_df[join_key + TARGET_COLS], on=join_key, how="left")

        for tgt in TARGET_COLS:
            nan_n = sub[tgt].isnull().sum()
            if nan_n:
                logger.warning("COORD: '%s' %d NaN → train ort. %.4f", tgt, nan_n, train_means[tgt])
                sub[tgt] = sub[tgt].fillna(train_means[tgt])
        return sub

    # ── ID sütununu bul ───────────────────────────────────────────────────────
    id_col = next(
        (c for c in sample_sub.columns
         if c.lower() in ("id", "row_id", "rowid", "sample_id")),
        sample_sub.columns[0],
    )
    logger.info("ID sütunu: '%s'", id_col)

    # pred_df'e ID üret: "ID_001", "ID_002", ... (sıralı, 3 basamak)
    pred_df = pred_df.copy().reset_index(drop=True)
    pred_df["_id_num"] = range(1, len(pred_df) + 1)

    # ── Senaryo A: Wide (ID + TA + EC + DRP) ─────────────────────────────────
    if fmt == "wide":
        logger.info("Submission format: WIDE (ID + hedef sütunları)")

        # sample_sub'daki hedef sütunlarını normalize et
        target_col_map: Dict[str, str] = {}  # sub_col → canonical target
        for col in sample_sub.columns:
            if col == id_col:
                continue
            canonical = _normalize_target_col(col)
            if canonical:
                target_col_map[col] = canonical

        logger.info("Hedef sütun eşlemesi: %s", target_col_map)

        # ID'nin sayısal kısmını çıkar ve pred_df ile eşleştir
        # Örn: "ID_001" → 1,  "row_001" → 1,  "1" → 1
        def _extract_num(id_val: str) -> int:
            digits = "".join(c for c in str(id_val) if c.isdigit())
            return int(digits) if digits else -1

        sub["_id_num"] = sub[id_col].apply(_extract_num)

        # Left merge: şablon sırası korunur
        cols_needed = ["_id_num"] + TARGET_COLS
        merged = sub.merge(pred_df[cols_needed], on="_id_num", how="left",
                           suffixes=("_old", ""))

        # Sonuçları sub'a yaz
        for sub_col, canonical in target_col_map.items():
            if canonical in merged.columns:
                sub[sub_col] = merged[canonical].values
            nan_n = sub[sub_col].isnull().sum()
            if nan_n:
                logger.warning("WIDE: '%s' %d NaN → train ort. %.4f",
                               sub_col, nan_n, train_means[canonical])
                sub[sub_col] = sub[sub_col].fillna(train_means[canonical])

        sub = sub.drop(columns=["_id_num"], errors="ignore")
        return sub

    # ── Senaryo B: Long / Melted (ID + Value) ─────────────────────────────────
    if fmt == "long":
        logger.info("Submission format: LONG/MELTED (ID + Value)")

        value_col = next(
            (c for c in sample_sub.columns if c in _VALUE_COL_CANDIDATES),
            sample_sub.columns[-1],
        )
        logger.info("Value sütunu: '%s'", value_col)

        # pred_df'i eriyik forma getir
        pred_melted = pred_df.melt(
            id_vars=["_id_num"],
            value_vars=TARGET_COLS,
            var_name="_target_name",
            value_name="_pred_value",
        )

        # ID formatını anla: ID_001_Total_Alkalinity  veya  ID_001_TA
        # Önce ilk birkaç ID'ye bakarak pattern çıkar
        sample_ids = sample_sub[id_col].head(10).tolist()
        logger.info("Örnek ID'ler: %s", sample_ids[:5])

        # sub'daki ID'den sayı + hedef kısmını ayrıştır
        def _parse_sub_id(id_val: str):
            """(satır_no:int, hedef_canonical:str | None) döndürür."""
            s = str(id_val)
            num_part = "".join(c for c in s if c.isdigit())
            row_num  = int(num_part) if num_part else -1
            # hedef kısmı: sayısal olmayan kısım
            txt_part = s.lower()
            for alias_key, canonical in _TARGET_ALIASES.items():
                if alias_key.replace(" ", "_") in txt_part or \
                   alias_key.replace(" ", "") in txt_part:
                    return row_num, canonical
            return row_num, None

        sub["_row_num"], sub["_target_canonical"] = zip(
            *sub[id_col].apply(_parse_sub_id)
        )

        # pred_melted ile eşleştir
        pred_melted_key = pred_melted.rename(
            columns={"_id_num": "_row_num", "_target_name": "_target_canonical"}
        )
        merged = sub.merge(
            pred_melted_key[["_row_num", "_target_canonical", "_pred_value"]],
            on=["_row_num", "_target_canonical"],
            how="left",
        )
        sub[value_col] = merged["_pred_value"].values

        # Fallback: eşleşemeyen satırlar → hedef bazlı train ortalaması
        for i, row in sub.iterrows():
            if pd.isna(sub.at[i, value_col]):
                canonical = row.get("_target_canonical")
                fallback  = train_means.get(canonical, 0.0) if canonical else 0.0
                sub.at[i, value_col] = fallback
                logger.warning("LONG: ID=%s eşleşemedi → fallback=%.4f",
                               row[id_col], fallback)

        sub = sub.drop(columns=["_row_num", "_target_canonical"], errors="ignore")
        return sub

    raise ValueError(f"Bilinmeyen format: {fmt}")


def _save_submission_bulletproof(
    sample_sub_path: Path,
    pred_df:         pd.DataFrame,
    train_means:     Dict[str, float],
    output_dir:      Path,
) -> None:
    """
    Kurşun geçirmez Zindi submission üretici.

    1. Şablonu oku → format tespit et (coord / wide / long)
    2. Tahminleri hizala → eksik ID'leri train ort. ile doldur
    3. assert: şablon satır sayısı == çıktı satır sayısı, null == 0
    4. Kaydet
    """
    if not sample_sub_path.exists():
        # Fallback: data_dir içinde olası isimler
        candidates = [
            sample_sub_path.parent / "SampleSubmission.csv",
            sample_sub_path.parent / "sample_submission.csv",
            sample_sub_path.parent / "submission_template.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Submission şablonu bulunamadı: {sample_sub_path}\n"
                f"Denenen yollar: {[str(p) for p in candidates]}"
            )
        logger.warning("Şablon yolu bulunamadı → '%s' kullanılıyor.", found)
        sample_sub_path = found

    sample_sub = pd.read_csv(sample_sub_path)
    n_expected = len(sample_sub)
    logger.info("Şablon: %s | %d satır | sütunlar: %s",
                sample_sub_path.name, n_expected, sample_sub.columns.tolist())

    fmt = _detect_format(sample_sub)
    logger.info("Tespit edilen format: %s", fmt.upper())

    result = _build_submission(sample_sub, pred_df, train_means, fmt)

    # ── Nihai doğrulama ───────────────────────────────────────────────────────
    if len(result) != n_expected:
        raise RuntimeError(
            f"Submission satır sayısı yanlış: {len(result)} ≠ {n_expected}. "
            "Format eşleştirme başarısız — manuel kontrol gerekli."
        )

    # Sayısal sütunlarda null kontrolü (ID sütunu hariç)
    num_cols   = result.select_dtypes(include=[np.number]).columns.tolist()
    null_total = result[num_cols].isnull().sum().sum()
    if null_total > 0:
        logger.error("%d null kaldı → 0 ile dolduruldu (son savunma).", null_total)
        result[num_cols] = result[num_cols].fillna(0)

    path = output_dir / "submission_ensemble.csv"
    result.to_csv(path, index=False)
    logger.info(
        "✅ Submission kaydedildi: %s | %d satır | format=%s | null=0",
        path, len(result), fmt.upper(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANA EĞİTİM
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:          Path,
    output_dir:        Path,
    sample_sub_path:   Optional[Path] = None,
    n_splits:          int  = _CV_N_SPLITS,
    drp_log_transform: bool = False,
) -> Dict[str, float]:

    active = [(name, fn) for name, ok, fn in _MODELS if ok]
    if not active:
        raise RuntimeError("Hiçbir model kütüphanesi yüklü değil.")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Aktif modeller: %s", [n for n, _ in active])

    # Submission şablon yolu (varsayılan: data_dir/SampleSubmission.csv)
    if sample_sub_path is None:
        sample_sub_path = data_dir / "SampleSubmission.csv"

    # ── 1. Veri ───────────────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("ADIM 1/4 — Veri yükleniyor")
    logger.info("=" * 62)
    loader   = WaterQualityDataLoader(data_dir=data_dir, drp_log_transform=False)
    train_df = _add_seasonality(loader.load_train())
    test_df  = _add_seasonality(loader.load_test())

    logger.info("Train: %s | Test: %s | Features: %d",
                train_df.shape, test_df.shape, len(ALL_FEATURE_COLS))

    X_train = train_df[ALL_FEATURE_COLS].astype(np.float32)
    X_test  = test_df[ALL_FEATURE_COLS].astype(np.float32)

    # ── 2. GroupKFold ──────────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("ADIM 2/4 — GroupKFold(station_id)")
    logger.info("=" * 62)
    cv = WaterQualityGroupKFold(
        n_splits=n_splits, group_col="station_id",
        freeze_path=output_dir / "fold_indices.json",
    )
    folds = list(cv.split(train_df))
    cv.validate_no_leakage(train_df)

    # ── 3. OOF eğitimi ────────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("ADIM 3/4 — OOF (%d model × 3 hedef × %d fold)",
                len(active), n_splits)
    logger.info("=" * 62)

    oof_store:  Dict[str, Dict[str, np.ndarray]] = {n: {} for n, _ in active}
    test_store: Dict[str, Dict[str, np.ndarray]] = {n: {} for n, _ in active}
    per_model_rows: List[dict] = []
    oof_r2_map: Dict[str, Dict[str, float]]      = {n: {} for n, _ in active}

    for target in TARGET_COLS:
        tag    = _SHORT[target]
        y      = train_df[target].values.astype(np.float64)
        y_fit  = np.log1p(y) if (drp_log_transform and "Phosphorus" in target) else y
        is_log = drp_log_transform and "Phosphorus" in target

        logger.info("─── %s (%s) ─────────────────────────────────", target, tag)

        for model_name, run_fn in active:
            oof         = np.zeros(len(train_df))
            test_folds: List[np.ndarray] = []
            fold_r2s:   List[float]      = []

            for fi, (tr, val) in enumerate(folds):
                vp, tp, best = run_fn(
                    X_train.iloc[tr], y_fit[tr],
                    X_train.iloc[val], y_fit[val],
                    X_test, target,
                )
                if is_log:
                    vp, tp = np.expm1(vp), np.expm1(tp)
                oof[val] = vp
                test_folds.append(tp)
                fr2 = r2_score(y[val], vp)
                fold_r2s.append(fr2)
                logger.info("  [%s] %s Fold %d | best=%4d | R²=%+.4f",
                            model_name, tag, fi + 1, best, fr2)

            oof_r2 = r2_score(y, oof)
            oof_store[model_name][target]  = oof
            test_store[model_name][target] = np.mean(test_folds, axis=0)
            oof_r2_map[model_name][tag]    = oof_r2
            logger.info("  >>> [%s] %s  OOF R² = %+.4f <<<", model_name, tag, oof_r2)
            per_model_rows.append({
                "model":    model_name,
                "target":   tag,
                "oof_r2":   round(oof_r2, 5),
                "fold_r2s": str([round(x, 4) for x in fold_r2s]),
            })

    # ── 4. Ağırlıklı Ensemble + Kurşun Geçirmez Submission ───────────────────
    logger.info("=" * 62)
    logger.info("ADIM 4/4 — OOF-ağırlıklı ensemble + submission")
    logger.info("=" * 62)

    ens_test = _weighted_blend(test_store, active, oof_r2_map)
    ens_oof  = _weighted_blend(oof_store,  active, oof_r2_map)

    scores: Dict[str, float] = {}
    for target in TARGET_COLS:
        scores[_SHORT[target]] = r2_score(train_df[target].values, ens_oof[target])
    scores["mean_r2"] = float(np.mean([scores["TA"], scores["EC"], scores["DRP"]]))

    _print_summary(scores, active, per_model_rows)

    # Test koordinatları ile tahminleri bir araya getir
    pred_df = test_df[["Latitude", "Longitude", "Sample Date"]].copy().reset_index(drop=True)
    for target in TARGET_COLS:
        pred_df[target] = ens_test[target]

    train_means = {t: float(train_df[t].mean()) for t in TARGET_COLS}

    # Kurşun geçirmez submission
    _save_submission_bulletproof(
        sample_sub_path = sample_sub_path,
        pred_df         = pred_df,
        train_means     = train_means,
        output_dir      = output_dir,
    )

    # OOF + per model sonuçlar
    _save_oof(train_df, ens_oof, output_dir)
    _save_per_model(per_model_rows, output_dir)
    _log_experiment(scores, active, output_dir, drp_log_transform)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(scores, active, rows):
    sep = "─" * 62
    print(f"\n{sep}")
    print("  BİREYSEL MODEL OOF R²")
    print(f"  {'Model':12s}  {'TA':>8}  {'EC':>8}  {'DRP':>8}")
    print("  " + "-" * 40)
    for model_name, _ in active:
        d = {r["target"]: r["oof_r2"] for r in rows if r["model"] == model_name}
        print(f"  {model_name:12s}  {d.get('TA', float('nan')):+.4f}   "
              f"{d.get('EC', float('nan')):+.4f}   {d.get('DRP', float('nan')):+.4f}")
    print(sep)
    print("  OOF-AĞIRLIKLI ENSEMBLE R²")
    print(sep)
    for key, label in [("TA",      "Total Alkalinity         "),
                       ("EC",      "Electrical Conductance   "),
                       ("DRP",     "Dissolved Reactive Phos. "),
                       ("mean_r2", "ORTALAMA R²              ")]:
        val  = scores.get(key, 0.0)
        bar  = "█" * max(0, int((val + 1) * 15))
        flag = "  ← HEDEF" if key == "mean_r2" else ""
        print(f"  {label}  {val:+.4f}  {bar}{flag}")
    print(sep + "\n")


def _save_oof(train_df, ens_oof, output_dir):
    out = train_df[["Latitude", "Longitude", "Sample Date", "station_id"]
                   + TARGET_COLS].copy()
    for tgt, pred in ens_oof.items():
        out[f"{_SHORT[tgt]}_oof"] = pred
    out.to_csv(output_dir / "oof_predictions.csv", index=False)


def _save_per_model(rows, output_dir):
    path = output_dir / "results_per_model.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info("Bireysel model sonuçları: %s", path)


def _log_experiment(scores, active, output_dir, drp_log):
    path = output_dir.parent / "experiments" / "leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":     "+".join(n for n, _ in active),
        "cv":        "GroupKFold_station",
        "blend":     "OOF_weighted",
        "TA_r2":     round(scores.get("TA",  0), 5),
        "EC_r2":     round(scores.get("EC",  0), 5),
        "DRP_r2":    round(scores.get("DRP", 0), 5),
        "mean_r2":   round(scores.get("mean_r2", 0), 5),
        "LB_r2":     "",
        "notes":     f"bulletproof_sub|oof_weighted|tweedie|drp_log={drp_log}",
    }
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)
    logger.info("Leaderboard: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EY Su Kalitesi 2026 — Ensemble")
    p.add_argument("--data_dir",    type=Path, default=Path("data/raw"))
    p.add_argument("--output_dir",  type=Path, default=Path("outputs"))
    p.add_argument("--sample_sub",  type=Path, default=None,
                   help="Zindi SampleSubmission.csv yolu "
                        "(default: data_dir/SampleSubmission.csv)")
    p.add_argument("--n_splits",    type=int,  default=_CV_N_SPLITS)
    p.add_argument("--drp_log",     action="store_true")
    args = p.parse_args()

    train(
        data_dir          = args.data_dir,
        output_dir        = args.output_dir,
        sample_sub_path   = args.sample_sub,
        n_splits          = args.n_splits,
        drp_log_transform = args.drp_log,
    )
    sys.exit(0)
