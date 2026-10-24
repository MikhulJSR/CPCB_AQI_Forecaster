"""
End-to-End Training Pipeline for the Ridge Baseline Model.
Loads feature matrix, executes chronological split, fits scaler, tunes regularization alpha,
and serializes model artifacts.
"""

import logging
from pathlib import Path
import pandas as pd
import joblib

from src.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    DEFAULT_FORECAST_HORIZON,
    RIDGE_ALPHA_GRID,
)
from src.models.preprocessor import TemporalDataSplitter, LeakageFreeScaler
from src.models.ridge_baseline import RidgeBaselineModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_training(horizon: int = DEFAULT_FORECAST_HORIZON) -> RidgeBaselineModel:
    """Executes the complete baseline training workflow."""
    feature_file = PROCESSED_DATA_DIR / f"cpcb_delhi_features_h{horizon}.csv"
    assert feature_file.exists(), f"Feature file {feature_file} not found. Run feature builder first."

    logger.info("Loading feature matrix from %s...", feature_file)
    df = pd.read_csv(feature_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_col = f"target_pm25_h{horizon}"
    metadata_cols = ["station", "timestamp", "latitude", "longitude", target_col, f"target_grap_iv_h{horizon}"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    logger.info("Identified %d predictor features. Target: %s", len(feature_cols), target_col)

    # Step 1: Chronological temporal splitting
    splitter = TemporalDataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df, feature_cols=feature_cols, target_col=target_col)

    # Step 2: Leakage-free standardization (fit strictly on training window)
    scaler = LeakageFreeScaler()
    X_train_scaled = scaler.fit_transform(splits["X_train"])
    X_val_scaled = scaler.transform(splits["X_val"])
    X_test_scaled = scaler.transform(splits["X_test"])

    # Step 3: Hyperparameter search over candidate alphas
    baseline_model = RidgeBaselineModel()
    best_alpha = baseline_model.tune_regularization(
        X_train=X_train_scaled,
        y_train=splits["y_train"],
        X_val=X_val_scaled,
        y_val=splits["y_val"],
        alphas=RIDGE_ALPHA_GRID
    )

    # Step 4: Fit final model on combined Train + Validation with optimal alpha
    logger.info("Fitting final baseline model on combined Train + Validation with alpha=%.2f...", best_alpha)
    import numpy as np
    X_combined = np.vstack([X_train_scaled, X_val_scaled])
    y_combined = np.concatenate([splits["y_train"], splits["y_val"]])

    baseline_model.fit(X_combined, y_combined, feature_names=feature_cols)

    # Step 5: Save model and scaler artifacts
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"ridge_baseline_h{horizon}.joblib"
    scaler_path = models_dir / f"scaler_h{horizon}.joblib"

    baseline_model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved fitted scaler artifact to %s", scaler_path)

    # Step 6: Display top learned physical coefficients
    df_coef = baseline_model.get_feature_coefficients()
    print("\n" + "=" * 60)
    print("TOP 10 LEARNED FEATURE WEIGHTS (PHYSICAL VERIFICATION):")
    print("=" * 60)
    print(df_coef.head(10).to_string(index=False))
    print("=" * 60)

    return baseline_model


if __name__ == "__main__":
    run_baseline_training()
