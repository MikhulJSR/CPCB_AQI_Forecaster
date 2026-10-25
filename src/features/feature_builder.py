"""
Feature Engineering and Target Alignment Pipeline for CPCB_AQI_Forecaster.
Constructs autoregressive lags, 24-hour rolling window statistics, decomposed wind vectors,
cyclical calendar encodings, and aligns multi-horizon forecasting targets.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    PRIMARY_TARGET,
    DEFAULT_FORECAST_HORIZON,
    REGULATORY_GRAP_IV_THRESHOLD,
    AUTOREGRESSIVE_LAGS,
    ROLLING_WINDOWS,
    HORIZONS,
)
from src.features.spatial_graph import AtmosphericSpatialGraph

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Production feature engineering engine for multi-station spatio-temporal telemetry."""

    def __init__(
        self,
        target_horizon_hours: int = DEFAULT_FORECAST_HORIZON,
        autoregressive_lags: List[int] = AUTOREGRESSIVE_LAGS,
        rolling_windows: List[int] = ROLLING_WINDOWS,
    ):
        self.horizon = target_horizon_hours
        self.lags = autoregressive_lags
        self.windows = rolling_windows

    @staticmethod
    def decompose_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose polar wind speed (m/s) and wind direction (degrees) into orthogonal
        Cartesian U (East-West) and V (North-South) wind vectors.
        """
        df_out = df.copy()
        if "wind_speed" in df_out.columns and "wind_direction" in df_out.columns:
            # Convert degrees to radians
            rad = np.deg2rad(df_out["wind_direction"].values)
            ws = df_out["wind_speed"].values

            # Meteorological convention: U is eastward, V is northward
            df_out["wind_u"] = -ws * np.sin(rad)
            df_out["wind_v"] = -ws * np.cos(rad)

            # Ventilation Index Proxy = Wind Speed * Temperature
            if "temp" in df_out.columns:
                df_out["ventilation_index_proxy"] = ws * np.maximum(1.0, df_out["temp"].values)

        return df_out

    @staticmethod
    def encode_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform hour-of-day and day-of-year into continuous sine/cosine periodic curves.
        Avoids the discontinuous jump between 23:00 and 00:00.
        """
        df_out = df.copy()
        timestamps = pd.to_datetime(df_out["timestamp"])

        hour = timestamps.dt.hour.values
        df_out["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df_out["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        day_of_year = timestamps.dt.day_of_year.values
        df_out["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_out["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

        return df_out

    def build_autoregressive_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct past temporal lag features for PM2.5 strictly within each station partition.
        """
        df_out = df.copy()
        for lag in self.lags:
            col_name = f"pm25_lag_{lag}"
            if lag == 0:
                df_out[col_name] = df_out[PRIMARY_TARGET]
            else:
                df_out[col_name] = df_out.groupby("station")[PRIMARY_TARGET].shift(lag)
        return df_out

    def build_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean, standard deviation, min, and max over the past W hours.
        Uses min_periods to tolerate minor missing values while guarding against lookahead.
        """
        df_out = df.copy()
        for w in self.windows:
            grouped = df_out.groupby("station")[PRIMARY_TARGET]
            # Closed on 'left' or shifted by 0 means using past w hours including current t
            df_out[f"pm25_roll_mean_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w // 2).mean())
            df_out[f"pm25_roll_std_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w // 2).std())
            df_out[f"pm25_roll_min_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w // 2).min())
            df_out[f"pm25_roll_max_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w // 2).max())
        return df_out

    def align_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align future target values:
        - target_pm25: Continuous PM2.5 at t + horizon hours.
        - target_grap_iv: Binary regulatory emergency indicator (target > 250 ug/m3).
        """
        df_out = df.copy()
        target_col = f"target_pm25_h{self.horizon}"
        binary_target_col = f"target_grap_iv_h{self.horizon}"

        df_out[target_col] = df_out.groupby("station")[PRIMARY_TARGET].shift(-self.horizon)
        df_out[binary_target_col] = (df_out[target_col] > REGULATORY_GRAP_IV_THRESHOLD).astype(float)
        # Preserve NaN for unobserved future tails
        df_out.loc[df_out[target_col].isna(), binary_target_col] = np.nan

        return df_out

    def generate_features(self, df: pd.DataFrame, drop_incomplete: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute original Tier 1 feature engineering pipeline.
        Returns the feature DataFrame and the list of predictor feature names.
        """
        logger.info("Building features with %d-hour target horizon...", self.horizon)

        df_wind = self.decompose_wind_vectors(df)
        df_time = self.encode_cyclical_time(df_wind)
        df_lags = self.build_autoregressive_lags(df_time)
        df_roll = self.build_rolling_statistics(df_lags)
        df_aligned = self.align_targets(df_roll)

        feature_cols = [
            f"pm25_lag_{lag}" for lag in self.lags
        ] + [
            f"pm25_roll_mean_{w}" for w in self.windows
        ] + [
            f"pm25_roll_std_{w}" for w in self.windows
        ] + [
            f"pm25_roll_min_{w}" for w in self.windows
        ] + [
            f"pm25_roll_max_{w}" for w in self.windows
        ] + [
            "wind_u", "wind_v", "temp", "humidity", "ventilation_index_proxy",
            "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
            "no2", "so2", "co"
        ]

        feature_cols = [c for c in feature_cols if c in df_aligned.columns]
        target_col = f"target_pm25_h{self.horizon}"
        if drop_incomplete:
            initial_count = len(df_aligned)
            df_aligned = df_aligned.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
            logger.info("Dropped incomplete boundary rows: %d -> %d clean rows retained.",
                        initial_count, len(df_aligned))

        return df_aligned, feature_cols

    def generate_tier2_features(
        self,
        df: pd.DataFrame,
        horizons: List[int] = HORIZONS,
        drop_incomplete: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build complete Tier 2 feature matrix:
        - Ingests sanitized continuous telemetry.
        - Computes dynamic wind-directed spatial graph features (AtmosphericSpatialGraph).
        - Computes autoregressive lags, 24h rolling window statistics, wind vectors, cyclical time.
        - Aligns multi-horizon forecasting targets (t+24h and t+48h).
        """
        logger.info("Building Tier 2 features with spatial graph and multi-horizons %s...", horizons)

        # 1. Atmospheric spatial graph cross-station features
        graph = AtmosphericSpatialGraph()
        df_spatial = graph.compute_dynamic_upwind_features(df)

        # 2. Physical wind vector decomposition
        df_wind = self.decompose_wind_vectors(df_spatial)

        # 3. Cyclical time encoding
        df_time = self.encode_cyclical_time(df_wind)

        # 4. Autoregressive lags
        df_lags = self.build_autoregressive_lags(df_time)

        # 5. Rolling window statistics
        df_roll = self.build_rolling_statistics(df_lags)

        # 6. Align multi-horizon targets
        df_aligned = df_roll.copy()
        target_cols = []
        for h in horizons:
            t_col = f"target_pm25_h{h}"
            b_col = f"target_grap_iv_h{h}"
            df_aligned[t_col] = df_aligned.groupby("station")[PRIMARY_TARGET].shift(-h)
            df_aligned[b_col] = (df_aligned[t_col] > REGULATORY_GRAP_IV_THRESHOLD).astype(float)
            df_aligned.loc[df_aligned[t_col].isna(), b_col] = np.nan
            target_cols.append(t_col)

        # Assemble list of Tier 2 predictor feature columns
        feature_cols = [
            f"pm25_lag_{lag}" for lag in self.lags
        ] + [
            f"pm25_roll_mean_{w}" for w in self.windows
        ] + [
            f"pm25_roll_std_{w}" for w in self.windows
        ] + [
            f"pm25_roll_min_{w}" for w in self.windows
        ] + [
            f"pm25_roll_max_{w}" for w in self.windows
        ] + [
            "pm25_upwind_current", "pm25_upwind_lag3", "upwind_advection_flux",
            "wind_u", "wind_v", "temp", "humidity", "ventilation_index_proxy",
            "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
            "no2", "so2", "co"
        ]

        feature_cols = [c for c in feature_cols if c in df_aligned.columns]

        if drop_incomplete:
            initial_count = len(df_aligned)
            # Require all features and at least the 24h target to be non-null
            df_aligned = df_aligned.dropna(subset=feature_cols + [target_cols[0]]).reset_index(drop=True)
            logger.info("Tier 2 feature matrix: %d -> %d clean rows retained.", initial_count, len(df_aligned))

        return df_aligned, feature_cols
