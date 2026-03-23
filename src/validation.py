"""
src/validation.py
=================
Station_id bazlı spatial GroupKFold.

ROLLBACK (CTO kararı): Strict temporal split veri açlığına yol açtı
(7500 → 1500 satır, R²=0.19). Yarışma hedefi spatial generalization;
temporal overlap bu veri setinde yapısal — kaldırıldı.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Spatial sızıntı tespit edildiğinde fırlatılır."""
    pass


class BaseSplitStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]: ...
    @abstractmethod
    def validate_no_leakage(self, df: pd.DataFrame) -> None: ...


class WaterQualityGroupKFold(BaseSplitStrategy):
    """
    Station_id bazlı spatial GroupKFold.

    Aynı istasyon hem train hem val'a karışmaz.
    Temporal overlap bu veri setinde yapısal (her istasyon 2011-2015 verir);
    temporal kısıtlama data starvation'a yol açtığından kaldırıldı.

    Parameters
    ----------
    n_splits    : int   Fold sayısı (default: 5)
    group_col   : str   İstasyon sütunu (default: "station_id")
    strict_mode : bool  True → aynı nehir de bölünmez
    river_col   : str   strict_mode nehir sütunu (default: "river_id")
    freeze_path : Path  Fold indeksleri JSON yolu; None → yazma
    """

    def __init__(
        self,
        n_splits:    int            = 5,
        group_col:   str            = "station_id",
        strict_mode: bool           = False,
        river_col:   str            = "river_id",
        freeze_path: Optional[Path] = None,
    ) -> None:
        self.n_splits    = n_splits
        self.group_col   = group_col
        self.strict_mode = strict_mode
        self.river_col   = river_col
        self.freeze_path = freeze_path

        self._kf = GroupKFold(n_splits=n_splits)
        self._fold_indices: List[Tuple[List[int], List[int]]] = []

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Train/val indeks çiftleri üret.

        Raises
        ------
        ValueError   : group_col eksikse
        LeakageError : Aynı istasyon hem train hem val'da ise
        """
        self._check_cols(df)
        groups  = df[self.group_col].values
        X_dummy = np.zeros(len(df))
        self._fold_indices = []

        for fold, (tr_idx, val_idx) in enumerate(
            self._kf.split(X_dummy, groups=groups)
        ):
            self._assert_no_station_leak(groups, tr_idx, val_idx, fold)
            if self.strict_mode:
                self._assert_no_river_leak(df, tr_idx, val_idx, fold)

            self._fold_indices.append((tr_idx.tolist(), val_idx.tolist()))
            logger.info("Fold %d | train: %d | val: %d",
                        fold + 1, len(tr_idx), len(val_idx))
            yield tr_idx, val_idx

        if self.freeze_path is not None:
            self._save_indices()

    def validate_no_leakage(self, df: pd.DataFrame) -> None:
        """
        Kaydedilmiş fold indekslerini retrospektif denetle.

        Raises
        ------
        RuntimeError : split() çağrılmamışsa
        LeakageError : Sızıntı tespit edilirse
        """
        if not self._fold_indices:
            raise RuntimeError(
                "validate_no_leakage() çağrısından önce split() çalıştırılmalı."
            )
        self._check_cols(df)
        groups = df[self.group_col].values
        for fold, (tr_idx, val_idx) in enumerate(self._fold_indices):
            tr_arr, val_arr = np.array(tr_idx), np.array(val_idx)
            self._assert_no_station_leak(groups, tr_arr, val_arr, fold)
            if self.strict_mode:
                self._assert_no_river_leak(df, tr_arr, val_arr, fold)

        logger.info("✅ Spatial sızıntı testi geçildi — %d fold temiz.", len(self._fold_indices))

    def load_frozen_indices(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._fold_indices = [(d["train"], d["val"]) for d in raw["folds"]]
        logger.info("Frozen indeksler yüklendi: %s", path)

    @property
    def n_folds(self) -> int:
        return len(self._fold_indices)

    def _check_cols(self, df: pd.DataFrame) -> None:
        missing = [self.group_col] if self.group_col not in df.columns else []
        if self.strict_mode and self.river_col not in df.columns:
            missing.append(self.river_col)
        if missing:
            raise ValueError(f"Eksik sütunlar: {missing}")

    def _assert_no_station_leak(self, groups, tr_idx, val_idx, fold):
        overlap = set(groups[tr_idx]) & set(groups[val_idx])
        if overlap:
            raise LeakageError(
                f"[SPATIAL LEAKAGE] Fold {fold+1}: "
                f"{len(overlap)} istasyon hem train hem val'da → {list(overlap)[:5]}"
            )

    def _assert_no_river_leak(self, df, tr_idx, val_idx, fold):
        overlap = (set(df.iloc[tr_idx][self.river_col])
                   & set(df.iloc[val_idx][self.river_col]))
        if overlap:
            raise LeakageError(
                f"[RIVER LEAKAGE] Fold {fold+1}: "
                f"{len(overlap)} nehir hem train hem val'da → {list(overlap)[:5]}"
            )

    def _save_indices(self) -> None:
        assert self.freeze_path is not None
        self.freeze_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_splits":  self.n_splits,
            "group_col": self.group_col,
            "folds": [{"fold": i+1, "train": tr, "val": val}
                      for i, (tr, val) in enumerate(self._fold_indices)],
        }
        with open(self.freeze_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Fold indeksleri kaydedildi: %s", self.freeze_path)


class StrictRiverKFold(WaterQualityGroupKFold):
    """strict_mode=True kısayolu."""
    def __init__(self, n_splits: int = 5, **kwargs) -> None:
        super().__init__(n_splits=n_splits, strict_mode=True, **kwargs)
