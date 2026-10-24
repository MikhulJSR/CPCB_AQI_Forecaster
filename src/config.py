"""
Central configuration module for CPCB_AQI_Forecaster.
Defines project paths, regulatory thresholds, feature schemas, and modeling constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

# CPCB Delhi Monitoring Station Constants
TARGET_CITY = "Delhi"
PRIMARY_TARGET = "pm25"
REGULATORY_GRAP_IV_THRESHOLD = 250.0  # PM2.5 in ug/m3 (Severe/Severe+ emergency threshold)

# Forecasting Horizons (hours ahead)
DEFAULT_FORECAST_HORIZON = 24
HORIZONS = [24, 48]

# Autoregressive and Rolling Feature Parameters
AUTOREGRESSIVE_LAGS = [0, 1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [24]  # 24-hour rolling window stats

# Core Criteria Pollutants & Exogenous Meteorology
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
METEOROLOGY = ["temp", "humidity", "wind_speed", "wind_direction"]

# Model Hyperparameters for Baseline
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RANDOM_STATE = 42
