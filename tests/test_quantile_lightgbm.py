"""
Unit tests for Multi-Horizon Quantile LightGBM Forecaster.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.quantile_lightgbm import QuantileLightGBMForecaster


def test_quantile_lightgbm_fit_and_monotonicity():
    """Verify fitting, prediction shapes, and non-crossing monotonicity (P10 <= P50 <= P90)."""
    np.random.seed(42)
    n_samples = 120
    X = pd.DataFrame({
        "lag_24": np.random.uniform(50.0, 300.0, n_samples),
        "wind_speed": np.random.uniform(0.5, 8.0, n_samples),
        "temp": np.random.uniform(10.0, 35.0, n_samples)
    })
    y_24 = X["lag_24"] * 0.9 - X["wind_speed"] * 4.0 + np.random.normal(0, 15.0, n_samples)
    y_48 = X["lag_24"] * 0.85 - X["wind_speed"] * 3.5 + np.random.normal(0, 20.0, n_samples)

    forecaster = QuantileLightGBMForecaster(
        horizons=[24, 48],
        quantiles=[0.10, 0.50, 0.90],
        n_estimators=20,
        min_child_samples=5
    )

    forecaster.fit(X, y_train_dict={24: y_24.values, 48: y_48.values})
    assert forecaster.is_fitted

    preds_24 = forecaster.predict_quantiles(X, horizon=24, enforce_monotonicity=True)
    assert "p10" in preds_24 and "p50" in preds_24 and "p90" in preds_24
    assert len(preds_24["p10"]) == n_samples

    # Verify strict mathematical non-crossing: P10 <= P50 <= P90
    assert np.all(preds_24["p10"] <= preds_24["p50"] + 1e-5)
    assert np.all(preds_24["p50"] <= preds_24["p90"] + 1e-5)

    # Verify 48h horizon also works
    preds_48 = forecaster.predict_quantiles(X, horizon=48, enforce_monotonicity=True)
    assert np.all(preds_48["p10"] <= preds_48["p50"] + 1e-5)
    assert np.all(preds_48["p50"] <= preds_48["p90"] + 1e-5)


def test_feature_importances_extraction():
    """Verify that feature importances are extracted and sorted."""
    np.random.seed(42)
    X = pd.DataFrame({
        "f_strong": np.random.uniform(10, 100, 80),
        "f_noise": np.random.uniform(0, 1, 80)
    })
    y = X["f_strong"] * 2.5 + np.random.normal(0, 2, 80)

    forecaster = QuantileLightGBMForecaster(horizons=[24], n_estimators=25, min_child_samples=5)
    forecaster.fit(X, y_train_dict={24: y.values})

    df_imp = forecaster.get_feature_importances(horizon=24, alpha=0.50)
    assert len(df_imp) == 2
    assert "feature" in df_imp.columns and "importance" in df_imp.columns
    assert df_imp.iloc[0]["feature"] == "f_strong"


def test_serialization_round_trip(tmp_path):
    """Verify serialization and deserialization of Quantile LightGBM models."""
    X = pd.DataFrame({"feat": [10.0, 20.0, 30.0, 40.0, 50.0]})
    y = np.array([12.0, 22.0, 28.0, 42.0, 49.0])

    forecaster = QuantileLightGBMForecaster(horizons=[24], n_estimators=10, min_child_samples=2)
    forecaster.fit(X, y_train_dict={24: y})

    artifact_path = tmp_path / "quantile_lgbm.joblib"
    forecaster.save(artifact_path)

    loaded = QuantileLightGBMForecaster.load(artifact_path)
    assert loaded.is_fitted
    assert loaded.horizons == [24]
    p_orig = forecaster.predict_quantiles(X, horizon=24)
    p_loaded = loaded.predict_quantiles(X, horizon=24)
    np.testing.assert_array_almost_equal(p_orig["p50"], p_loaded["p50"])
