"""
Multi-Horizon Quantile LightGBM Forecaster for CPCB_AQI_Forecaster.
Implements gradient boosted decision trees optimizing the asymmetric Pinball Loss
to output calibrated prediction intervals (P10, P50, P90) across multiple forecast horizons.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from src.config import HORIZONS, RANDOM_STATE

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES = [0.10, 0.50, 0.90]


class QuantileLightGBMForecaster:
    """Production multi-horizon probabilistic forecaster using Quantile LightGBM."""

    def __init__(
        self,
        horizons: List[int] = HORIZONS,
        quantiles: List[float] = DEFAULT_QUANTILES,
        num_leaves: int = 31,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
        min_child_samples: int = 20,
        random_state: int = RANDOM_STATE
    ):
        self.horizons = horizons
        self.quantiles = sorted(quantiles)
        self.hyperparams = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        # Nested dictionary of models: self.models[horizon][alpha]
        self.models: Dict[int, Dict[float, lgb.LGBMRegressor]] = {h: {} for h in self.horizons}
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_dict: Dict[int, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "QuantileLightGBMForecaster":
        """
        Fit distinct quantile tree ensembles for each target horizon and each quantile alpha.
        """
        self.feature_names_ = list(feature_names) if feature_names is not None else list(X_train.columns)
        X_mat = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

        logger.info(
            "Fitting Multi-Horizon Quantile LightGBM across %d horizons and %d quantiles (%d total models)...",
            len(self.horizons), len(self.quantiles), len(self.horizons) * len(self.quantiles)
        )

        for h in self.horizons:
            assert h in y_train_dict, f"Missing target array for horizon h={h}"
            y_h = y_train_dict[h]

            # Filter out any unaligned NaNs in target
            valid_mask = ~np.isnan(y_h)
            X_valid = X_mat[valid_mask]
            y_valid = y_h[valid_mask]

            for alpha in self.quantiles:
                logger.info("Training LightGBM [Horizon: t+%dh | Quantile: P%02d] on %d samples...",
                            h, int(alpha * 100), len(y_valid))
                
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    **self.hyperparams
                )
                model.fit(X_valid, y_valid)
                self.models[h][alpha] = model

        self.is_fitted = True
        logger.info("All Multi-Horizon Quantile LightGBM models successfully trained.")
        return self

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
        enforce_monotonicity: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic quantile forecasts for a specified horizon.
        Returns:
            {"p10": array, "p50": array, "p90": array}
        If enforce_monotonicity is True, prevents quantile crossing by monotonic sorting.
        """
        assert self.is_fitted, "Forecaster must be fitted before generating predictions"
        assert horizon in self.models, f"Horizon {horizon} not fitted in this model"

        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        raw_preds = {}

        for alpha in self.quantiles:
            model = self.models[horizon][alpha]
            pred = model.predict(X_mat)
            raw_preds[alpha] = pred

        # Convert to matrix of shape (n_samples, n_quantiles)
        pred_matrix = np.column_stack([raw_preds[a] for a in self.quantiles])

        # Enforce non-crossing: q10 <= q50 <= q90
        if enforce_monotonicity:
            pred_matrix = np.sort(pred_matrix, axis=1)

        # Build clean output dictionary
        output = {}
        for idx, alpha in enumerate(self.quantiles):
            key = f"p{int(alpha * 100)}"
            output[key] = np.round(pred_matrix[:, idx], 2)

        return output

    def get_feature_importances(self, horizon: int = 24, alpha: float = 0.50) -> pd.DataFrame:
        """
        Extract feature importances (gain / split) for a specific horizon and quantile.
        """
        assert self.is_fitted, "Model must be fitted to inspect feature importances"
        assert horizon in self.models and alpha in self.models[horizon], "Invalid horizon or alpha"

        model = self.models[horizon][alpha]
        importances = model.feature_importances_

        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: Path) -> None:
        """Serialize complete multi-horizon quantile forecaster to disk."""
        assert self.is_fitted, "Cannot save unfitted model"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models,
            "horizons": self.horizons,
            "quantiles": self.quantiles,
            "hyperparams": self.hyperparams,
            "feature_names": self.feature_names_,
        }, filepath)
        logger.info("Saved Multi-Horizon Quantile LightGBM artifact to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "QuantileLightGBMForecaster":
        """Load serialized multi-horizon quantile forecaster from disk."""
        data = joblib.load(filepath)
        instance = cls(
            horizons=data["horizons"],
            quantiles=data["quantiles"],
            **{k: v for k, v in data["hyperparams"].items() if k not in ["random_state", "verbose", "n_jobs"]}
        )
        instance.models = data["models"]
        instance.is_fitted = True
        instance.feature_names_ = data.get("feature_names", [])
        logger.info("Loaded Multi-Horizon Quantile LightGBM from %s", filepath)
        return instance
