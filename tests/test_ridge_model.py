"""
Unit tests for Ridge Baseline Model.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.ridge_baseline import RidgeBaselineModel


def test_ridge_fit_and_predict():
    """Verify that RidgeBaseline fits and outputs predictions of correct shape."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 50.0 + np.random.randn(100) * 0.1

    model = RidgeBaselineModel(alpha=1.0)
    model.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])

    assert model.is_fitted
    preds = model.predict(X)
    assert preds.shape == (100,)
    # R2 should be high for this simple synthetic signal
    from sklearn.metrics import r2_score
    assert r2_score(y, preds) > 0.90


def test_regularization_tuning():
    """Verify that tune_regularization selects an alpha from the candidate grid."""
    np.random.seed(42)
    X_train = np.random.randn(80, 4)
    y_train = 3.0 * X_train[:, 0] + np.random.randn(80) * 0.5
    X_val = np.random.randn(20, 4)
    y_val = 3.0 * X_val[:, 0] + np.random.randn(20) * 0.5

    model = RidgeBaselineModel()
    candidates = [0.01, 1.0, 100.0]
    best_alpha = model.tune_regularization(X_train, y_train, X_val, y_val, alphas=candidates)

    assert best_alpha in candidates
    assert len(model.tuning_history_) == 3


def test_model_save_and_load(tmp_path):
    """Verify that model state and coefficients serialize and deserialize identically."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])

    model = RidgeBaselineModel(alpha=5.0)
    model.fit(X, y, feature_names=["feature_a", "feature_b"])

    artifact_path = tmp_path / "test_ridge.joblib"
    model.save(artifact_path)

    loaded_model = RidgeBaselineModel.load(artifact_path)
    assert loaded_model.is_fitted
    assert loaded_model.alpha == 5.0
    assert loaded_model.feature_names_ == ["feature_a", "feature_b"]
    np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))
