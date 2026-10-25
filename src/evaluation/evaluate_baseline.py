"""
Evaluation Pipeline Runner for the Ridge Baseline Model.
Loads test partition and serialized model, executes regression and regulatory GRAP evaluation,
runs residual diagnostics, and generates a structured report.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    DEFAULT_FORECAST_HORIZON,
    REGULATORY_GRAP_IV_THRESHOLD,
)
from src.models.preprocessor import TemporalDataSplitter
from src.models.ridge_baseline import RidgeBaselineModel
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_evaluation(horizon: int = DEFAULT_FORECAST_HORIZON) -> Dict[str, Any]:
    """Runs full evaluation suite on the unseen test set."""
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / f"ridge_baseline_h{horizon}.joblib"
    scaler_path = models_dir / f"scaler_h{horizon}.joblib"

    assert model_path.exists(), f"Model {model_path} not found. Run train_baseline.py first."
    assert scaler_path.exists(), f"Scaler {scaler_path} not found. Run train_baseline.py first."

    # Load feature matrix and test split
    feature_file = PROCESSED_DATA_DIR / f"cpcb_delhi_features_h{horizon}.csv"
    df = pd.read_csv(feature_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_col = f"target_pm25_h{horizon}"
    metadata_cols = ["station", "timestamp", "latitude", "longitude", target_col, f"target_grap_iv_h{horizon}"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    splitter = TemporalDataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df, feature_cols=feature_cols, target_col=target_col)

    test_df = splits["test_df"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Load artifacts and generate predictions
    model = RidgeBaselineModel.load(model_path)
    scaler = joblib.load(scaler_path)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # 24-hour diurnal baseline reference (pm25_lag_24)
    y_diurnal = test_df["pm25_lag_24"].values if "pm25_lag_24" in test_df.columns else None

    # Compute metrics
    evaluator = ModelEvaluator(grap_threshold=REGULATORY_GRAP_IV_THRESHOLD)
    cont_metrics = evaluator.compute_continuous_metrics(y_test, y_pred, y_diurnal)
    grap_metrics = evaluator.compute_grap_exceedance_metrics(y_test, y_pred)
    diag_summary = evaluator.compute_residual_diagnostics(test_df, y_test, y_pred)

    # Save row-by-row prediction artifacts for inspection
    test_export = test_df.copy()
    test_export["y_pred_pm25_h24"] = np.round(y_pred, 2)
    test_export["residual"] = np.round(y_test - y_pred, 2)
    test_export["abs_error"] = np.round(np.abs(y_test - y_pred), 2)
    export_cols = [
        "station", "timestamp", "pm25_lag_0", "pm25_lag_24", "temp", "wind_speed",
        target_col, "y_pred_pm25_h24", "residual", "abs_error"
    ]
    export_cols = [c for c in export_cols if c in test_export.columns]
    pred_path = PROCESSED_DATA_DIR / f"baseline_test_predictions_h{horizon}.csv"
    test_export[export_cols].to_csv(pred_path, index=False)
    logger.info("Saved row-by-row predictions to %s", pred_path)

    # Generate Markdown Report
    report_content = f"""# Baseline Evaluation Report: Tier 1 Regularized Autoregressive Model (Ridge)

## 1. Executive Summary
* **Model:** Ridge Regression with L2 Regularization (Optimal $\\lambda = {model.alpha:.2f}$)
* **Forecast Horizon:** {horizon} Hours Ahead ($t+24\\text{{h}}$)
* **Evaluation Partition:** Chronological Unseen Test Set ({len(test_df)} observations across Delhi CAAQMS stations)
* **Date Range:** {test_df['timestamp'].min()} to {test_df['timestamp'].max()}

---

## 2. Continuous Regression Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | **{cont_metrics['mae']:.2f} $\\mu\\text{{g/m}}^3$** | Average absolute deviation on unseen future test period. |
| **RMSE** | **{cont_metrics['rmse']:.2f} $\\mu\\text{{g/m}}^3$** | Penalizes severe peak misses. |
| **$R^2$ Score** | **{cont_metrics['r2']:.4f}** | Variance explained by the linear model. |
| **sMAPE** | **{cont_metrics['smape_pct']:.2f}%** | Symmetric Mean Absolute Percentage Error. |
| **Diurnal Baseline MAE** | **{cont_metrics.get('diurnal_baseline_mae', 'N/A')} $\\mu\\text{{g/m}}^3$** | 24-hour diurnal seasonal persistence benchmark. |
| **MASE** | **{cont_metrics.get('mase', 'N/A')}** | Mean Absolute Scaled Error (< 1.0 proves model beats persistence). |

---

## 3. Regulatory Decision Performance (GRAP Stage IV Emergency: $>250\\ \\mu\\text{{g/m}}^3$)
| Confusion Matrix Element | Count | Operational Impact |
| :--- | :--- | :--- |
| **True Positives (Hits)** | **{grap_metrics['true_positives_hits']}** | Emergency accurately anticipated $\\ge 24\\text{{h}}$ in advance. |
| **False Positives (False Alarms)** | **{grap_metrics['false_positives_false_alarms']}** | Unnecessary economic restriction cost. |
| **False Negatives (Misses)** | **{grap_metrics['false_negatives_misses']}** | Acute public health emergency without preparation. |
| **True Negatives** | **{grap_metrics['true_negatives']}** | Normal conditions accurately maintained. |

### Decision Metrics:
* **Lead-Time Exceedance Recall:** **{grap_metrics['recall'] * 100:.1f}%**
* **Precision:** **{grap_metrics['precision'] * 100:.1f}%**
* **Critical Success Index (CSI / Threat Score):** **{grap_metrics['csi_threat_score']:.4f}**

---

## 4. Residual Diagnostics: Atmospheric Stagnation Analysis
Performance breakdown across wind speed regimes:

{diag_summary.to_markdown(index=False) if not diag_summary.empty else 'No wind data available for grouping.'}

---

## 5. Scientific Gap Analysis: Why We Must Advance to Tier 2 (Quantile LightGBM)
1. **The Stagnation Blindspot:** Under stagnant wind conditions ($<1.5\\ \\text{{m/s}}$), the linear baseline exhibits increased residual bias because atmospheric dilution collapses exponentially rather than linearly.
2. **Deterministic Constraint:** Ridge outputs a single point estimate without calibrated uncertainty intervals ($P_{{10}}, P_{{90}}$).
3. **Spatial Transport:** Ridge treats stations independently without tracking upwind-to-downwind plume advection vectors.
"""

    report_path = PROJECT_ROOT / "docs" / "baseline_evaluation_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved baseline evaluation report to %s", report_path)

    print("\n" + "=" * 60)
    print("BASELINE TEST SET RESULTS:")
    print("=" * 60)
    print(f"MAE:  {cont_metrics['mae']:.2f} ug/m3 (vs Diurnal Baseline: {cont_metrics.get('diurnal_baseline_mae', 'N/A')} ug/m3)")
    print(f"MASE: {cont_metrics.get('mase', 'N/A')} (< 1.0 proves ML beats seasonal persistence)")
    print(f"R²:   {cont_metrics['r2']:.4f}")
    print(f"GRAP Stage IV Recall:    {grap_metrics['recall'] * 100:.1f}%")
    print(f"GRAP CSI (Threat Score): {grap_metrics['csi_threat_score']:.4f}")
    print("=" * 60)

    return {
        "continuous": cont_metrics,
        "grap": grap_metrics,
        "report_path": str(report_path)
    }


if __name__ == "__main__":
    run_baseline_evaluation()
