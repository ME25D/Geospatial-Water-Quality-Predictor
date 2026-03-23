"""
src/features.py
===============
Feature engineering katmanı.

FIX-1  TemporalLagTransformer: out.loc[X.index] ile index hizalama korundu.
FIX-2  DRPTransformer kaldırıldı → train.py'de TransformedTargetRegressor.
FIX-4  SatelliteTimeAlignmentFilter: >15 gün eski uydu verisi NaN.
NEW    SeasonalityTransformer: month → sin/cos döngüsel kodlama (ROLLBACK FIX-3).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

_SPECTRAL_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# YENİ — SEASONALITY TRANSFORMER  (CTO FIX-3)
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalityTransformer(BaseEstimator, TransformerMixin):
    """
    Tarih sütunundan mevsimsellik özellikleri türetir.

    Doğrusal month (1-12) yerine döngüsel sin/cos kodlama kullanılır:
      - month_sin = sin(2π × month / 12)
      - month_cos = cos(2π × month / 12)

    Bu kodlama Aralık (12) ile Ocak (1)'ın komşu olduğunu modele söyler.
    Su kalitesi güçlü mevsimsel desen gösterir (yağış, sıcaklık, buharlaşma).

    Parameters
    ----------
    date_col : str  Tarih sütunu (default: "Sample Date", dayfirst=True)
    drop_raw : bool True → ham month sütununu sonuçtan çıkar (default: False)
    """

    def __init__(
        self,
        date_col: str  = "Sample Date",
        drop_raw: bool = False,
    ) -> None:
        self.date_col = date_col
        self.drop_raw = drop_raw

    def fit(self, X: pd.DataFrame, y=None) -> "SeasonalityTransformer":
        if self.date_col not in X.columns:
            raise ValueError(f"'{self.date_col}' sütunu bulunamadı.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        month, month_sin, month_cos sütunlarını ekler.

        Raises
        ------
        NotFittedError : fit() çağrılmamışsa
        """
        check_is_fitted(self, ["is_fitted_"])
        out   = X.copy()
        dates = pd.to_datetime(out[self.date_col], dayfirst=True, errors="coerce")

        out["month"]     = dates.dt.month
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

        if self.drop_raw:
            out = out.drop(columns=["month"])

        return out


# ─────────────────────────────────────────────────────────────────────────────
# SATELLITE TIME ALIGNMENT FILTER  (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteTimeAlignmentFilter(BaseEstimator, TransformerMixin):
    """
    Uydu görüntüsü ölçüm tarihinden >max_delta_days gün eskiyse bantları NaN yap.

    Mevcut EY veri setinde 'image_date' sütunu yok → pasif mod (uyarı loglar).
    Kendi extraction pipeline'ında bu sütun eklenince otomatik aktif olur.

    Parameters
    ----------
    satellite_date_col : str   Uydu görüntü tarihi sütunu
    sample_date_col    : str   Ölçüm tarihi sütunu
    feature_cols       : list  Maskelenecek spektral sütunlar
    max_delta_days     : int   Tolerans eşiği (default: 15)
    """

    def __init__(
        self,
        satellite_date_col: str       = "image_date",
        sample_date_col:    str       = "Sample Date",
        feature_cols:       List[str] = None,
        max_delta_days:     int       = 15,
    ) -> None:
        self.satellite_date_col = satellite_date_col
        self.sample_date_col    = sample_date_col
        self.feature_cols       = feature_cols or [
            "nir", "green", "swir16", "swir22", "NDMI", "MNDWI"
        ]
        self.max_delta_days = max_delta_days

    def fit(self, X: pd.DataFrame, y=None) -> "SatelliteTimeAlignmentFilter":
        self.active_ = self.satellite_date_col in X.columns
        if not self.active_:
            logger.warning(
                "SatelliteTimeAlignmentFilter: '%s' sütunu yok — pasif mod.",
                self.satellite_date_col,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, ["active_"])
        if not self.active_:
            return X

        out        = X.copy()
        img_dt     = pd.to_datetime(out[self.satellite_date_col], errors="coerce")
        sample_dt  = pd.to_datetime(out[self.sample_date_col], dayfirst=True, errors="coerce")
        stale      = (sample_dt - img_dt).abs().dt.days > self.max_delta_days

        if stale.sum() > 0:
            logger.warning("%d satırda uydu >%d gün eski → NaN.",
                           stale.sum(), self.max_delta_days)
            mask_cols = [c for c in self.feature_cols if c in out.columns]
            out.loc[stale, mask_cols] = np.nan
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL INDEX TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

_SPECTRAL_INDEX_FORMULAS: Dict[str, Tuple[str, str]] = {
    "NDVI":  ("NIR",   "Red"),
    "NDWI":  ("Green", "NIR"),
    "MNDWI": ("Green", "SWIR1"),
    "NDTI":  ("Red",   "Green"),
    "NDBI":  ("SWIR1", "NIR"),
    "NDMI":  ("NIR",   "SWIR1"),
}


class SpectralIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Landsat bantlarından normalize spektral indeks hesaplar.

    OCP: Yeni indeks → _SPECTRAL_INDEX_FORMULAS sözlüğüne ekle, sınıf değişmez.

    Parameters
    ----------
    band_cols : dict  Bant adı → sütun adı  (örn: {"NIR": "nir", ...})
    indices   : list  Hesaplanacak indeks adları; None → tümü
    eps       : float Sıfıra bölme koruyucusu
    """

    def __init__(
        self,
        band_cols: Dict[str, str],
        indices:   Optional[List[str]] = None,
        eps:       float = _SPECTRAL_EPS,
    ) -> None:
        self.band_cols = band_cols
        self.indices   = indices or list(_SPECTRAL_INDEX_FORMULAS.keys())
        self.eps       = eps

    def fit(self, X: pd.DataFrame, y=None) -> "SpectralIndexTransformer":
        available = set(self.band_cols)
        self.computable_indices_: List[str] = []
        for name in self.indices:
            if name not in _SPECTRAL_INDEX_FORMULAS:
                continue
            a, b = _SPECTRAL_INDEX_FORMULAS[name]
            if a in available and b in available:
                self.computable_indices_.append(name)
            else:
                logger.warning("%s atlandı — eksik bantlar: %s / %s", name, a, b)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, ["computable_indices_"])
        out = X.copy()
        for name in self.computable_indices_:
            a_band, b_band = _SPECTRAL_INDEX_FORMULAS[name]
            a = out[self.band_cols[a_band]].to_numpy(dtype=float)
            b = out[self.band_cols[b_band]].to_numpy(dtype=float)
            out[name] = (a - b) / (a + b + self.eps)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL LAG TRANSFORMER  (FIX-1: index hizalama korundu)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalLagTransformer(BaseEstimator, TransformerMixin):
    """
    İstasyon bazlı lag ve rolling mean/std özellikleri.

    FIX-1: sort_values() sonrası out.loc[X.index] ile orijinal satır
    sırası geri yüklenir — X/y eşleşmesi bozulmaz.

    Parameters
    ----------
    lag_cols        : list  Lag uygulanacak sütunlar
    lag_months      : list  Gecikme adımları (ay)
    rolling_windows : list  Hareketli ortalama pencereleri (ay)
    group_col       : str   İstasyon sütunu
    date_col        : str   Tarih sütunu
    """

    def __init__(
        self,
        lag_cols:        List[str],
        lag_months:      List[int] = None,
        rolling_windows: List[int] = None,
        group_col: str = "station_id",
        date_col:  str = "date",
    ) -> None:
        self.lag_cols        = lag_cols
        self.lag_months      = lag_months      or [1, 2, 3]
        self.rolling_windows = rolling_windows or [3, 6]
        self.group_col       = group_col
        self.date_col        = date_col

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalLagTransformer":
        missing = [c for c in self.lag_cols + [self.group_col, self.date_col]
                   if c not in X.columns]
        if missing:
            raise ValueError(f"Eksik sütunlar: {missing}")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, ["is_fitted_"])
        out = X.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])
        out = out.sort_values([self.group_col, self.date_col])

        for col in self.lag_cols:
            grp = out.groupby(self.group_col)[col]
            for lag in self.lag_months:
                out[f"{col}_lag{lag}"] = grp.shift(lag)
            for w in self.rolling_windows:
                out[f"{col}_roll{w}_mean"] = grp.transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).mean()
                )
                out[f"{col}_roll{w}_std"] = grp.transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).std()
                )

        return out.loc[X.index]   # FIX-1: orijinal index geri yüklendi
