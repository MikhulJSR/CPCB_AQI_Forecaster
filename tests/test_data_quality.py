"""
Unit tests for Data Quality and Sanitization Pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_quality.sanitizer import DataSanitizer


def test_physical_bounds_validation():
    """Verify that negative concentrations and impossible spikes are purged."""
    data = pd.DataFrame({
        "station": ["Anand Vihar"] * 4,
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=4, freq="1h"),
        "pm25": [-15.0, 150.0, 1200.0, 200.0],  # -15 and 1200 are physically impossible
        "pm10": [50.0, 200.0, 300.0, 180.0],    # 180 is less than pm25 (200.0) -> inconsistency
        "humidity": [120.0, 50.0, -5.0, 60.0]   # 120 and -5 are impossible
    })

    sanitizer = DataSanitizer()
    clean_df, report = sanitizer.validate_physical_bounds(data)

    assert np.isnan(clean_df.loc[0, "pm25"])
    assert np.isnan(clean_df.loc[2, "pm25"])
    assert clean_df.loc[1, "pm25"] == 150.0
    assert clean_df.loc[3, "pm10"] >= clean_df.loc[3, "pm25"]
    assert np.isnan(clean_df.loc[0, "humidity"])
    assert np.isnan(clean_df.loc[2, "humidity"])


def test_stuck_sensor_detection():
    """Verify that a frozen sensor reporting constant values for >= 6h is identified and purged."""
    # 7 identical values in a row (e.g. exactly 180.0)
    data = pd.DataFrame({
        "station": ["ITO"] * 10,
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=10, freq="1h"),
        "pm25": [150.0, 160.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 195.0]
    })

    sanitizer = DataSanitizer(stuck_sensor_threshold_hours=6)
    clean_df, stuck_count = sanitizer.detect_stuck_sensors(data)

    assert stuck_count >= 1
    # Check that frozen values were converted to NaN
    assert np.isnan(clean_df.loc[8, "pm25"])


def test_short_gap_vs_long_gap_imputation():
    """Verify that gaps <= 3 hours are imputed, but gaps > 3 hours retain NaNs."""
    # 2-hour gap (should be imputed) followed by 5-hour gap (should remain NaN)
    pm25_series = [100.0, np.nan, np.nan, 130.0, np.nan, np.nan, np.nan, np.nan, np.nan, 200.0]
    data = pd.DataFrame({
        "station": ["Punjabi Bagh"] * len(pm25_series),
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=len(pm25_series), freq="1h"),
        "pm25": pm25_series
    })

    sanitizer = DataSanitizer(max_interpolation_gap_hours=3)
    clean_df, report = sanitizer.impute_short_gaps(data)

    # 2-hour gap between 100.0 and 130.0 should be filled linearly: 110.0, 120.0
    assert not np.isnan(clean_df.loc[1, "pm25"])
    assert not np.isnan(clean_df.loc[2, "pm25"])
    assert clean_df.loc[1, "pm25"] == 110.0
    assert clean_df.loc[2, "pm25"] == 120.0

    # 5-hour gap should NOT be completely filled (retains NaNs beyond limit=3)
    assert np.isnan(clean_df.loc[7, "pm25"])
    assert np.isnan(clean_df.loc[8, "pm25"])
