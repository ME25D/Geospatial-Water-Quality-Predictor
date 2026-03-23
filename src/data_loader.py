"""
src/data_loader.py
==================
EY 2026 Su Kalitesi — Veri yükleme katmanı.

KRITIK FIX (Zindi submission hatası):
  Test setinden HİÇBİR SATIR SİLİNMEZ.
  - Merge: her zaman how="left" (200 satır → 200 satır)
  - Imputation: train'de fit, test'te transform; pet dahil tüm sütunlar
  - Fallback: imputer sonrası kalan NaN → .fillna(0)
  - self._template: orijinal 200 satırlık şablon saklanır;
    train.py buradan alır, pozisyonel atama yaparak hiç satır kaçırmaz
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────
_COORD_PRECISION      = 6
_LANDSAT_FEATURE_COLS = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]
_TC_FEATURE_COLS      = ["pet"]
_JOIN_KEY             = ["Latitude", "Longitude", "Sample Date"]

TARGET_COLS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

FEATURE_COLS = (
    ["Latitude", "Longitude", "year", "month"]
    + _LANDSAT_FEATURE_COLS
    + _TC_FEATURE_COLS
)

# imputation uygulanacak tüm sayısal özellik sütunları
_IMPUTE_COLS = _LANDSAT_FEATURE_COLS + _TC_FEATURE_COLS


class BaseDataLoader(ABC):
    @abstractmethod
    def load_train(self) -> pd.DataFrame: ...
    @abstractmethod
    def load_test(self) -> pd.DataFrame: ...


class WaterQualityDataLoader(BaseDataLoader):
    """
    EY 2026 Su Kalitesi yarışması veri loader'ı.

    Satır Güvencesi
    ---------------
    - load_train() → 9319 satır (training dataset satır sayısı)
    - load_test()  → submission_template.csv satır sayısı (200)
      Hiçbir koşulda bu sayıların altına düşülmez.

    Parameters
    ----------
    data_dir          : Path  CSV dosyalarının klasörü
    impute_strategy   : str   SimpleImputer stratejisi ("median")
    drp_log_transform : bool  DRP hedefine log1p uygula (default: False)
    """

    def __init__(
        self,
        data_dir:          Path,
        impute_strategy:   str  = "median",
        drp_log_transform: bool = False,
    ) -> None:
        self.data_dir          = Path(data_dir)
        self.impute_strategy   = impute_strategy
        self.drp_log_transform = drp_log_transform
        self._imputer: Optional[SimpleImputer] = None
        self._template: Optional[pd.DataFrame] = None  # orijinal submission şablonu
        self._train_target_means: dict = {}             # fallback NaN doldurucu

    # ── Public API ────────────────────────────────────────────────────────────

    def load_train(self) -> pd.DataFrame:
        """Eğitim setini yükle, birleştir, ön işle."""
        logger.info("Eğitim verisi yükleniyor...")
        wq = self._csv("water_quality_training_dataset.csv")
        ls = self._csv("landsat_features_training.csv")
        tc = self._csv("terraclimate_features_training.csv")

        n_before = len(wq)
        df = self._merge(wq, ls, tc)
        assert len(df) == n_before, (
            f"Eğitim merge sonrası satır kaybı: {n_before} → {len(df)}"
        )

        df = self._engineer(df, is_train=True)

        # Hedef ortalamalarını sakla (test fallback için)
        for col in TARGET_COLS:
            if col in df.columns:
                self._train_target_means[col] = float(df[col].mean())

        logger.info("Eğitim seti hazır: %s", df.shape)
        return df

    def load_test(self) -> pd.DataFrame:
        """
        Test setini yükle. load_train() önce çağrılmış olmalı.

        Satır Güvencesi: submission_template.csv satır sayısı korunur.
        Hiçbir merge veya imputation satır düşürmez.
        """
        if self._imputer is None:
            raise RuntimeError(
                "load_test() öncesinde load_train() çağrılmalı."
            )

        logger.info("Test verisi yükleniyor...")
        # Şablonu sakla — train.py buradan alır
        self._template = self._csv("submission_template.csv")
        n_expected     = len(self._template)

        ls = self._csv("landsat_features_validation.csv")
        tc = self._csv("terraclimate_features_validation.csv")

        # Merge: şablonun JOIN sütunlarını temel al → satır sayısı sabit
        base = self._template[_JOIN_KEY].copy()
        df   = self._merge(base, ls, tc)

        # KRITIK: satır sayısı değişmediyse devam, değiştiyse hata
        if len(df) != n_expected:
            raise RuntimeError(
                f"Test merge sonrası satır kayması: "
                f"beklenen={n_expected}, gerçek={len(df)}"
            )

        df = self._engineer(df, is_train=False)

        # Son savunma: hâlâ NaN kaldıysa 0 ile doldur (satır kaybı olmaz)
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isnull().any():
                logger.warning("Test: '%s' sütununda kalan NaN → 0 ile dolduruldu.", col)
                df[col] = df[col].fillna(0)

        assert len(df) == n_expected, (
            f"load_test() çıktısı {len(df)} satır — beklenen {n_expected}"
        )
        logger.info("Test seti hazır: %s", df.shape)
        return df

    def get_submission_template(self) -> pd.DataFrame:
        """
        Orijinal submission şablonunu döndür.
        load_test() sonrası çağrılmalı (self._template dolu olmalı).
        """
        if self._template is not None:
            return self._template.copy()
        # Fallback: diskten oku
        return self._csv("submission_template.csv")

    # ── Private: IO ───────────────────────────────────────────────────────────

    def _csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        return pd.read_csv(path)

    # ── Private: merge (her zaman LEFT join — satır düşmez) ──────────────────

    def _merge(self, base: pd.DataFrame, ls: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
        ls_use = [c for c in _JOIN_KEY + _LANDSAT_FEATURE_COLS if c in ls.columns]
        tc_use = [c for c in _JOIN_KEY + _TC_FEATURE_COLS      if c in tc.columns]

        df = base.merge(ls[ls_use], on=_JOIN_KEY, how="left")
        df = df.merge(tc[tc_use],  on=_JOIN_KEY, how="left")
        return df

    # ── Private: feature engineering ─────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df.copy()
        df = self._parse_dates(df)
        df = self._assign_station_id(df)
        df = self._impute(df, is_train)

        if is_train and self.drp_log_transform:
            col = "Dissolved Reactive Phosphorus"
            if col in df.columns:
                if (df[col] < 0).any():
                    raise ValueError("DRP negatif değer; log1p uygulanamaz.")
                df[col] = np.log1p(df[col])
                logger.info("DRP log1p uygulandı.")

        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"]  = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce")
        df["year"]  = df["date"].dt.year.fillna(0).astype(int)
        df["month"] = df["date"].dt.month.fillna(0).astype(int)
        return df

    def _assign_station_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """(Latitude, Longitude) → deterministik MD5 station_id."""
        def _hash(lat: float, lon: float) -> str:
            key = f"{round(lat, _COORD_PRECISION)}_{round(lon, _COORD_PRECISION)}"
            return "STN_" + hashlib.md5(key.encode()).hexdigest()[:8]

        df["station_id"] = df.apply(
            lambda r: _hash(r["Latitude"], r["Longitude"]), axis=1
        )
        logger.info("station_id türetildi — benzersiz: %d", df["station_id"].nunique())
        return df

    def _impute(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Landsat + TerraClimate NaN → SimpleImputer.
        Train'de fit, test'te transform. Satır düşürmez.
        """
        cols = [c for c in _IMPUTE_COLS if c in df.columns]
        if not cols:
            return df

        if is_train:
            self._imputer = SimpleImputer(strategy=self.impute_strategy)
            df[cols] = self._imputer.fit_transform(df[cols])
        else:
            # transform: imputer train istatistikleriyle doldurur
            df[cols] = self._imputer.transform(df[cols])

        remaining = df[cols].isnull().sum().sum()
        if remaining:
            logger.warning("Imputation sonrası %d NaN kaldı → 0 ile dolduruldu.", remaining)
            df[cols] = df[cols].fillna(0)

        return df
