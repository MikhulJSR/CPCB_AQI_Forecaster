"""
Leakage-Free Temporal Splitting and Preprocessing Engine.
Enforces chronological time boundaries, purged rolling-window cross-validation,
and strict training-only standardization for linear models.
"""

import logging
from typing import List, Tuple, Dict, Any, Generator
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """Partitions multi-station spatio-temporal datasets along strict chronological boundaries."""

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        timestamp_col: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Chronologically splits DataFrame into train, validation, and test partitions.
        Guarantees: max(train_time) < min(val_time) < min(test_time).
        """
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        unique_times = df_sorted[timestamp_col].drop_duplicates().sort_values().values
        n_times = len(unique_times)

        train_cutoff_idx = int(n_times * self.train_ratio)
        val_cutoff_idx = int(n_times * (self.train_ratio + self.val_ratio))

        train_cutoff_time = unique_times[train_cutoff_idx]
        val_cutoff_time = unique_times[val_cutoff_idx]

        train_mask = df_sorted[timestamp_col] < train_cutoff_time
        val_mask = (df_sorted[timestamp_col] >= train_cutoff_time) & (df_sorted[timestamp_col] < val_cutoff_time)
        test_mask = df_sorted[timestamp_col] >= val_cutoff_time

        train_df = df_sorted[train_mask].reset_index(drop=True)
        val_df = df_sorted[val_mask].reset_index(drop=True)
        test_df = df_sorted[test_mask].reset_index(drop=True)

        logger.info(
            "Temporal Split Complete | Train: %d rows (%s to %s) | Val: %d rows (%s to %s) | Test: %d rows (%s to %s)",
            len(train_df), train_df[timestamp_col].min(), train_df[timestamp_col].max(),
            len(val_df), val_df[timestamp_col].min(), val_df[timestamp_col].max(),
            len(test_df), test_df[timestamp_col].min(), test_df[timestamp_col].max(),
        )

        return {
            "X_train": train_df[feature_cols].copy(),
            "y_train": train_df[target_col].values,
            "X_val": val_df[feature_cols].copy(),
            "y_val": val_df[target_col].values,
            "X_test": test_df[feature_cols].copy(),
            "y_test": test_df[target_col].values,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }


class LeakageFreeScaler:
    """Wrapper around StandardScaler ensuring parameters (mean, std) are computed strictly on train data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "LeakageFreeScaler":
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted, "Scaler must be fitted on training data before transforming"
        return self.scaler.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


class PurgedTimeSeriesSplit:
    """
    Expanding-window walk-forward time-series cross-validation generator with an embargo gap.
    Prevents autocorrelation leakage between adjacent train and validation folds.
    """

    def __init__(self, n_splits: int = 4, embargo_hours: int = 24):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours

    def split(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        timestamps = pd.to_datetime(df[timestamp_col])
        unique_times = timestamps.drop_duplicates().sort_values().values
        n_times = len(unique_times)

        # Minimum initial training chunk is 40% of time steps
        min_train_size = int(n_times * 0.40)
        remaining = n_times - min_train_size
        val_chunk_size = remaining // self.n_splits

        for i in range(self.n_splits):
            train_end_idx = min_train_size + i * val_chunk_size
            val_start_idx = train_end_idx + self.embargo_hours  # Embargo gap
            val_end_idx = min(train_end_idx + (i + 1) * val_chunk_size, n_times)

            if val_start_idx >= val_end_idx:
                continue

            train_end_time = unique_times[train_end_idx]
            val_start_time = unique_times[val_start_idx]
            val_end_time = unique_times[val_end_idx - 1]

            train_indices = np.where(timestamps <= train_end_time)[0]
            val_indices = np.where((timestamps >= val_start_time) & (timestamps <= val_end_time))[0]

            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices
