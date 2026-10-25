"""
End-to-End Training Pipeline for Tier 2: Multi-Horizon Quantile LightGBM.
Loads Tier 2 feature matrix, executes chronological split, tunes tree hyperparameters,
fits multi-horizon models (24h and 48h for P10, P50, P90), and persists model artifacts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib

from src.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    HORIZONS,
    RANDOM_STATE,
)
from src.models.preprocessor import TemporalDataSplitter
from src.models.quantile_lightgbm import QuantileLightGBMForecaster, DEFAULT_QUANTILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Candidate hyperparameter grid for tuning
LGBM_CANDIDATE_CONFIGS = [
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "name": "Shallow (Leaves: 15, Depth: 4)"},
    {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "n_estimators": 150, "name": "Standard (Leaves: 31, Depth: 6)"},
    {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.03, "n_estimators": 200, "name": "Deep (Leaves: 63, Depth: 8)"},
]


def tune_lgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train_24: np.ndarray,
    X_val: pd.DataFrame,
    y_val_24: np.ndarray,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Search candidate LightGBM architectures against the validation partition.
    Selects the configuration minimizing validation median (P50) MAE.
    """
    best_config = LGBM_CANDIDATE_CONFIGS[0]
    lowest_val_mae = float("inf")
    tuning_records = []

    logger.info("Starting LightGBM hyperparameter search across %d configurations...", len(LGBM_CANDIDATE_CONFIGS))

    for cfg in LGBM_CANDIDATE_CONFIGS:
        name = cfg["name"]
        forecaster = QuantileLightGBMForecaster(
            horizons=[24],
            quantiles=[0.50],
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            min_child_samples=20,
            random_state=RANDOM_STATE
        )
        forecaster.fit(X_train, {24: y_train_24}, feature_names=feature_cols)

        preds_val = forecaster.predict_quantiles(X_val, horizon=24)["p50"]
        val_mae = float(mean_absolute_error(y_val_24, preds_val))

        tuning_records.append({
            "config": name,
            "val_mae": round(val_mae, 2),
            "num_leaves": cfg["num_leaves"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"]
        })
        logger.info("Config: %-35s | Validation P50 MAE: %6.2f ug/m3", name, val_mae)

        if val_mae < lowest_val_mae:
            lowest_val_mae = val_mae
            best_config = cfg

    logger.info("Optimal LightGBM Architecture Selected: %s (Val MAE: %.2f ug/m3)",
                best_config["name"], lowest_val_mae)
    return best_config


def run_tier2_training() -> QuantileLightGBMForecaster:
    """Executes the complete Tier 2 training workflow."""
    feature_file = PROCESSED_DATA_DIR / "cpcb_delhi_features_tier2.csv"
    assert feature_file.exists(), f"Tier 2 feature file {feature_file} not found. Run Step 3 first."

    logger.info("Loading Tier 2 feature matrix from %s...", feature_file)
    df = pd.read_csv(feature_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_cols = [f"target_pm25_h{h}" for h in HORIZONS]
    grap_cols = [f"target_grap_iv_h{h}" for h in HORIZONS]
    metadata_cols = ["station", "timestamp", "latitude", "longitude"] + target_cols + grap_cols
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    logger.info("Identified %d predictor features (including spatial graph inputs).", len(feature_cols))

    # Step 1: Chronological temporal splitting
    splitter = TemporalDataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df, feature_cols=feature_cols, target_col="target_pm25_h24")

    X_train = splits["X_train"]
    y_train_24 = splits["y_train"]
    X_val = splits["X_val"]
    y_val_24 = splits["y_val"]

    # Target 48h arrays
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    y_train_48 = train_df["target_pm25_h48"].values
    y_val_48 = val_df["target_pm25_h48"].values

    # Step 2: Tune hyperparameters on 24h validation set
    best_cfg = tune_lgbm_hyperparameters(X_train, y_train_24, X_val, y_val_24, feature_cols)

    # Step 3: Combine Train + Validation for final model fitting
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined_dict = {
        24: np.concatenate([y_train_24, y_val_24]),
        48: np.concatenate([y_train_48, y_val_48]),
    }

    logger.info("Fitting final Multi-Horizon Quantile LightGBM on %d combined observations...", len(X_combined))
    final_forecaster = QuantileLightGBMForecaster(
        horizons=HORIZONS,
        quantiles=DEFAULT_QUANTILES,
        num_leaves=best_cfg["num_leaves"],
        max_depth=best_cfg["max_depth"],
        learning_rate=best_cfg["learning_rate"],
        n_estimators=best_cfg["n_estimators"],
        min_child_samples=20,
        random_state=RANDOM_STATE
    )
    final_forecaster.fit(X_combined, y_combined_dict, feature_names=feature_cols)

    # Step 4: Persist model artifact
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "quantile_lgbm_multi_horizon.joblib"
    final_forecaster.save(model_path)

    # Step 5: Feature Importances
    print("\n" + "=" * 65)
    print("TOP 10 FEATURE IMPORTANCES (HORIZON: t+24h | QUANTILE: P50 MEDIAN):")
    print("=" * 65)
    df_imp_24 = final_forecaster.get_feature_importances(horizon=24, alpha=0.50)
    print(df_imp_24.head(10).to_string(index=False))
    print("=" * 65)

    print("\n" + "=" * 65)
    print("TOP 10 FEATURE IMPORTANCES (HORIZON: t+48h | QUANTILE: P50 MEDIAN):")
    print("=" * 65)
    df_imp_48 = final_forecaster.get_feature_importances(horizon=48, alpha=0.50)
    print(df_imp_48.head(10).to_string(index=False))
    print("=" * 65)

    # Highlight spatial graph feature ranks
    spatial_features = ["pm25_upwind_current", "pm25_upwind_lag3", "upwind_advection_flux"]
    print("\n--- Atmospheric Spatial Graph Feature Contributions ---")
    for feat in spatial_features:
        rank_24 = df_imp_24[df_imp_24["feature"] == feat].index[0] + 1
        imp_24 = df_imp_24[df_imp_24["feature"] == feat]["importance"].values[0]
        rank_48 = df_imp_48[df_imp_48["feature"] == feat].index[0] + 1
        imp_48 = df_imp_48[df_imp_48["feature"] == feat]["importance"].values[0]
        print(f"Feature: {feat:<25} | 24h Rank: #{rank_24:<2} (Gain: {imp_24:>4}) | 48h Rank: #{rank_48:<2} (Gain: {imp_48:>4})")

    return final_forecaster


if __name__ == "__main__":
    run_tier2_training()
