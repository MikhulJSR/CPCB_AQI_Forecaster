"""
Unit tests for Temporal Preprocessing, Purged Cross-Validation, and Scaling.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.preprocessor import TemporalDataSplitter, LeakageFreeScaler, PurgedTimeSeriesSplit


def test_temporal_data_splitter_boundaries():
    """Verify strict chronological monotonicity: max(train) < min(val) < min(test)."""
    n_hours = 100
    times = pd.date_range("2026-10-01 00:00:00", periods=n_hours, freq="1h")
    data = pd.DataFrame({
        "timestamp": times,
        "feature_1": np.random.randn(n_hours),
        "target": np.random.randn(n_hours)
    })

    splitter = TemporalDataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(data, feature_cols=["feature_1"], target_col="target")

    max_train_time = splits["train_df"]["timestamp"].max()
    min_val_time = splits["val_df"]["timestamp"].min()
    max_val_time = splits["val_df"]["timestamp"].max()
    min_test_time = splits["test_df"]["timestamp"].min()

    assert max_train_time < min_val_time
    assert max_val_time < min_test_time
    assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == n_hours


def test_leakage_free_scaler():
    """Verify that scaler parameters are calculated strictly from training fold."""
    # Training fold with mean=100.0, Test fold with mean=500.0
    X_train = pd.DataFrame({"f1": [90.0, 100.0, 110.0]})
    X_test = pd.DataFrame({"f1": [490.0, 500.0, 510.0]})

    scaler = LeakageFreeScaler()
    scaled_train = scaler.fit_transform(X_train)

    # Scaled training mean should be 0.0
    assert np.isclose(scaled_train.mean(), 0.0, atol=1e-5)
    # Scaler mean must strictly equal X_train mean (100.0), unaffected by X_test
    assert scaler.scaler.mean_[0] == 100.0

    scaled_test = scaler.transform(X_test)
    # 500 transformed with train mean (100) and std (8.16) must be positive and far from 0
    assert scaled_test.mean() > 40.0


def test_purged_time_series_split():
    """Verify that expanding window cross-validation respects the embargo gap."""
    n_hours = 200
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=n_hours, freq="1h"),
        "val": np.arange(n_hours)
    })

    cv = PurgedTimeSeriesSplit(n_splits=3, embargo_hours=12)
    splits = list(cv.split(data))

    assert len(splits) >= 2
    for train_idx, val_idx in splits:
        train_max_time = data.loc[train_idx, "timestamp"].max()
        val_min_time = data.loc[val_idx, "timestamp"].min()

        # Validation must start AT LEAST 12 hours after training ends
        time_diff_hours = (val_min_time - train_max_time).total_seconds() / 3600.0
        assert time_diff_hours >= 12.0
