"""
Ridge Baseline Model Module for CPCB_AQI_Forecaster.
Implements L2-regularized linear autoregression, hyperparameter search over alpha (lambda),
and feature coefficient inspection.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config import RIDGE_ALPHA_GRID, RANDOM_STATE

logger = logging.getLogger(__name__)


class RidgeBaselineModel:
    """Production-grade Ridge regression baseline forecaster."""

    def __init__(self, alpha: float = 1.0, random_state: int = RANDOM_STATE):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.tuning_history_: List[Dict[str, float]] = []

    def tune_regularization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        alphas: Optional[List[float]] = None
    ) -> float:
        """
        Evaluate candidate regularization strengths (alphas) against the validation partition.
        Selects and sets the alpha that minimizes validation MAE.
        """
        alphas = alphas or RIDGE_ALPHA_GRID
        best_alpha = alphas[0]
        lowest_val_mae = float("inf")
        self.tuning_history_ = []

        logger.info("Starting Ridge regularization grid search over %d candidate alphas...", len(alphas))

        for a in alphas:
            candidate = Ridge(alpha=a, random_state=self.random_state)
            candidate.fit(X_train, y_train)

            train_preds = candidate.predict(X_train)
            val_preds = candidate.predict(X_val)

            train_mae = mean_absolute_error(y_train, train_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)

            self.tuning_history_.append({
                "alpha": a,
                "train_mae": round(train_mae, 4),
                "val_mae": round(val_mae, 4),
                "val_r2": round(val_r2, 4),
            })

            logger.info("Alpha: %8.2f | Train MAE: %6.2f | Val MAE: %6.2f | Val R²: %6.4f",
                        a, train_mae, val_mae, val_r2)

            if val_mae < lowest_val_mae:
                lowest_val_mae = val_mae
                best_alpha = a

        self.alpha = best_alpha
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        logger.info("Optimal Regularization Alpha Selected: %.2f (Validation MAE: %.2f)",
                    self.alpha, lowest_val_mae)
        return self.alpha

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RidgeBaselineModel":
        """Fit Ridge model on input array."""
        self.model.fit(X, y)
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate continuous point predictions."""
        assert self.is_fitted, "Model must be fitted before generating predictions"
        return self.model.predict(X)

    def get_feature_coefficients(self) -> pd.DataFrame:
        """
        Return DataFrame of learned weights for each feature, sorted by absolute importance.
        """
        assert self.is_fitted, "Model must be fitted to inspect coefficients"
        names = self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(len(self.model.coef_))]
        df_coef = pd.DataFrame({
            "feature": names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_)
        }).sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
        return df_coef

    def save(self, filepath: Path) -> None:
        """Persist model artifact to disk."""
        assert self.is_fitted, "Cannot save unfitted model"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "alpha": self.alpha,
            "feature_names": self.feature_names_,
            "tuning_history": self.tuning_history_
        }, filepath)
        logger.info("Saved Ridge baseline model artifact to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "RidgeBaselineModel":
        """Load persisted model artifact from disk."""
        data = joblib.load(filepath)
        instance = cls(alpha=data["alpha"])
        instance.model = data["model"]
        instance.is_fitted = True
        instance.feature_names_ = data.get("feature_names", [])
        instance.tuning_history_ = data.get("tuning_history", [])
        logger.info("Loaded Ridge baseline model from %s (alpha=%.2f)", filepath, instance.alpha)
        return instance
