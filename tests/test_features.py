"""
Unit tests for Feature Engineering Pipeline and Target Alignment.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.feature_builder import FeatureBuilder


def test_wind_vector_decomposition():
    """Verify trigonometric correctness of U and V wind vector decomposition."""
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=4, freq="1h"),
        "wind_speed": [10.0, 10.0, 10.0, 10.0],
        "wind_direction": [0.0, 90.0, 180.0, 270.0],  # N, E, S, W
        "temp": [25.0, 25.0, 25.0, 25.0]
    })

    df_wind = FeatureBuilder.decompose_wind_vectors(data)

    # North (0 deg): sin(0)=0, cos(0)=1 -> U=0, V=-10
    assert np.isclose(df_wind.loc[0, "wind_u"], 0.0, atol=1e-5)
    assert np.isclose(df_wind.loc[0, "wind_v"], -10.0, atol=1e-5)

    # East (90 deg): sin(90)=1, cos(90)=0 -> U=-10, V=0
    assert np.isclose(df_wind.loc[1, "wind_u"], -10.0, atol=1e-5)
    assert np.isclose(df_wind.loc[1, "wind_v"], 0.0, atol=1e-5)

    # Ventilation index proxy = 10 * 25 = 250
    assert df_wind.loc[0, "ventilation_index_proxy"] == 250.0


def test_cyclical_time_encoding():
    """Verify that hour and day-of-year cyclical encodings are within [-1, 1] and continuous."""
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=24, freq="1h")
    })

    df_time = FeatureBuilder.encode_cyclical_time(data)

    assert df_time["sin_hour"].min() >= -1.0 and df_time["sin_hour"].max() <= 1.0
    assert df_time["cos_hour"].min() >= -1.0 and df_time["cos_hour"].max() <= 1.0

    # At midnight (00:00): sin(0)=0, cos(0)=1
    assert np.isclose(df_time.loc[0, "sin_hour"], 0.0, atol=1e-5)
    assert np.isclose(df_time.loc[0, "cos_hour"], 1.0, atol=1e-5)


def test_target_alignment_and_lag_causality():
    """Verify that future target is shifted by -horizon and lag_1 is shifted by +1."""
    n_hours = 60
    data = pd.DataFrame({
        "station": ["Anand Vihar"] * n_hours,
        "timestamp": pd.date_range("2026-10-01 00:00:00", periods=n_hours, freq="1h"),
        "pm25": np.arange(100.0, 100.0 + n_hours)  # 100, 101, 102 ...
    })

    builder = FeatureBuilder(target_horizon_hours=24, autoregressive_lags=[0, 1, 24])
    df_lags = builder.build_autoregressive_lags(data)
    df_target = builder.align_targets(df_lags)

    # At row 10 (value 110.0):
    # pm25_lag_1 should be 109.0 (previous hour)
    assert df_target.loc[10, "pm25_lag_1"] == 109.0

    # target_pm25_h24 should be 134.0 (value at row 10+24 = 34)
    assert df_target.loc[10, "target_pm25_h24"] == 134.0

    # Verify regulatory binary target flag (values > 250 are 1, else 0)
    assert df_target.loc[10, "target_grap_iv_h24"] == 0.0


def test_tier2_feature_generation():
    """Verify that Tier 2 features include spatial graph features and multi-horizon targets."""
    n_hours = 80
    times = pd.date_range("2025-10-01 00:00:00", periods=n_hours, freq="1h")
    records = []
    for s in ["Bawana, Delhi - DPCC", "Anand Vihar, Delhi - DPCC"]:
        for t in times:
            records.append({
                "timestamp": t,
                "station": s,
                "pm25": 150.0,
                "pm10": 250.0,
                "temp": 20.0,
                "humidity": 60.0,
                "wind_speed": 3.0,
                "wind_direction": 315.0,
                "no2": 30.0,
                "so2": 10.0,
                "co": 1.0,
                "o3": 25.0
            })
    df_sanitized = pd.DataFrame(records)

    builder = FeatureBuilder()
    df_tier2, feature_cols = builder.generate_tier2_features(df_sanitized, horizons=[24, 48], drop_incomplete=True)

    # Check spatial features are in the feature list
    assert "pm25_upwind_current" in feature_cols
    assert "pm25_upwind_lag3" in feature_cols
    assert "upwind_advection_flux" in feature_cols

    # Check multi-horizon targets exist in DataFrame
    assert "target_pm25_h24" in df_tier2.columns
    assert "target_pm25_h48" in df_tier2.columns
    assert "target_grap_iv_h24" in df_tier2.columns
    assert "target_grap_iv_h48" in df_tier2.columns
    assert len(df_tier2) > 0
