"""
Evaluation and Diagnostic Module for CPCB_AQI_Forecaster.
Computes continuous regression metrics, regulatory GRAP exceedance scores (CSI, Precision, Recall),
and generates residual diagnostics against meteorological variables.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import REGULATORY_GRAP_IV_THRESHOLD

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation engine for air quality forecasting models."""

    def __init__(self, grap_threshold: float = REGULATORY_GRAP_IV_THRESHOLD):
        self.threshold = grap_threshold

    @staticmethod
    def compute_continuous_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_diurnal_baseline: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute continuous regression metrics: MAE, RMSE, R2, sMAPE, and MASE.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        # Symmetric Mean Absolute Percentage Error (sMAPE)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape = float(np.mean(np.where(denominator == 0, 0, np.abs(y_pred - y_true) / denominator)) * 100.0)

        metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "smape_pct": round(smape, 2),
        }

        # Mean Absolute Scaled Error (MASE) against diurnal persistence baseline
        if y_diurnal_baseline is not None:
            baseline_mae = float(mean_absolute_error(y_true, y_diurnal_baseline))
            mase = mae / (baseline_mae + 1e-8)
            metrics["mase"] = round(mase, 4)
            metrics["diurnal_baseline_mae"] = round(baseline_mae, 2)

        return metrics

    def compute_grap_exceedance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute regulatory decision metrics on the GRAP Stage IV emergency threshold (>250 ug/m3).
        Calculates Confusion Matrix, Precision, Recall, F1, and Critical Success Index (CSI / Threat Score).
        """
        y_true_binary = (y_true > self.threshold).astype(int)
        y_pred_binary = (y_pred > self.threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

        precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        # Critical Success Index (Threat Score) = Hits / (Hits + False Alarms + Misses)
        csi_denominator = tp + fp + fn
        csi = float(tp / csi_denominator) if csi_denominator > 0 else 0.0

        return {
            "threshold_ug_m3": self.threshold,
            "true_positives_hits": int(tp),
            "false_positives_false_alarms": int(fp),
            "false_negatives_misses": int(fn),
            "true_negatives": int(tn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "csi_threat_score": round(csi, 4),
        }

    @staticmethod
    def compute_residual_diagnostics(
        df_test: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze residuals (y_true - y_pred) grouped by meteorological regimes (e.g. wind speed bins).
        Reveals systematic under-prediction during atmospheric stagnation events.
        """
        residuals = y_true - y_pred
        df_diag = df_test.copy()
        df_diag["residual"] = residuals
        df_diag["abs_residual"] = np.abs(residuals)

        # Bin wind speed into Stagnant (<1.5 m/s), Moderate (1.5-3.0 m/s), High (>3.0 m/s)
        if "wind_speed" in df_diag.columns:
            df_diag["wind_regime"] = pd.cut(
                df_diag["wind_speed"],
                bins=[-np.inf, 1.5, 3.0, np.inf],
                labels=["Stagnant (<1.5 m/s)", "Moderate (1.5-3.0 m/s)", "Breezy (>3.0 m/s)"]
            )
            regime_summary = df_diag.groupby("wind_regime", observed=False).agg(
                count=("residual", "count"),
                mean_residual=("residual", "mean"),
                mae=("abs_residual", "mean"),
                std_residual=("residual", "std")
            ).reset_index()
            return regime_summary

        return pd.DataFrame()
