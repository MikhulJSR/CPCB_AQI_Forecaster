"""
Data Quality and Sanitization Module for CPCB CAAQMS Telemetry.
Enforces physical plausibility bounds, detects sensor anomalies/freezes,
regularizes time grids, and applies leakage-free imputation rules.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.config import PRIMARY_TARGET

logger = logging.getLogger(__name__)

# Regulatory and physical instrument limits
PHYSICAL_BOUNDS = {
    "pm25": (0.0, 1000.0),       # ug/m3
    "pm10": (0.0, 1500.0),       # ug/m3
    "no2": (0.0, 500.0),         # ug/m3
    "so2": (0.0, 500.0),         # ug/m3
    "co": (0.0, 50.0),           # mg/m3
    "o3": (0.0, 500.0),          # ug/m3
    "temp": (-5.0, 55.0),        # °C
    "humidity": (0.0, 100.0),    # %
    "wind_speed": (0.0, 45.0),   # m/s
    "wind_direction": (0.0, 360.0),  # degrees
}


class DataSanitizer:
    """Production data quality enforcement and sanitization engine."""

    def __init__(self, max_interpolation_gap_hours: int = 3, stuck_sensor_threshold_hours: int = 6):
        self.max_gap = max_interpolation_gap_hours
        self.stuck_threshold = stuck_sensor_threshold_hours

    def validate_physical_bounds(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Clip or set to NaN any values that violate physical instruments or atmospheric limits.
        Detects PM2.5 > PM10 violations (physical impossibility).
        """
        df_clean = df.copy()
        bounds_violations = {}

        for col, (lower, upper) in PHYSICAL_BOUNDS.items():
            if col in df_clean.columns:
                invalid_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                count = int(invalid_mask.sum())
                bounds_violations[col] = count
                if count > 0:
                    df_clean.loc[invalid_mask, col] = np.nan

        # Physical consistency check: PM2.5 cannot exceed PM10
        if "pm25" in df_clean.columns and "pm10" in df_clean.columns:
            inconsistent_mask = (df_clean["pm25"] > df_clean["pm10"]) & df_clean["pm25"].notna() & df_clean["pm10"].notna()
            pm_inconsistent_count = int(inconsistent_mask.sum())
            bounds_violations["pm25_exceeds_pm10"] = pm_inconsistent_count
            if pm_inconsistent_count > 0:
                # If PM2.5 exceeds PM10, PM10 reading was likely erroneous; set PM10 to PM2.5 * 1.2
                df_clean.loc[inconsistent_mask, "pm10"] = df_clean.loc[inconsistent_mask, "pm25"] * 1.2

        return df_clean, bounds_violations

    def detect_stuck_sensors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Detect frozen/stuck sensors where readings remain identically constant for >= threshold hours.
        In atmospheric monitoring, true ambient concentration never stays identical to 0.1 precision for 6+ hours.
        """
        df_clean = df.copy()
        total_stuck_flagged = 0

        for col in ["pm25", "pm10", "no2"]:
            if col not in df_clean.columns:
                continue

            for station, group in df_clean.groupby("station"):
                series = group[col]
                # Compare consecutive diffs
                diff = series.diff().abs()
                is_zero = (diff == 0.0) & series.notna()

                # Calculate consecutive identical run lengths
                run_length = is_zero.groupby((~is_zero).cumsum()).cumsum()
                stuck_mask = run_length >= (self.stuck_threshold - 1)
                stuck_indices = series[stuck_mask].index

                if len(stuck_indices) > 0:
                    total_stuck_flagged += len(stuck_indices)
                    df_clean.loc[stuck_indices, col] = np.nan

        return df_clean, total_stuck_flagged

    def regularize_time_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure every station has an uninterrupted, strictly hourly time index without gaps or duplicates.
        """
        regularized_stations = []

        for station_name, group in df.groupby("station"):
            # Deduplicate any exact timestamp entries
            dedup = group.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            dedup = dedup.set_index("timestamp")

            # Reindex to full continuous hourly grid
            full_idx = pd.date_range(start=dedup.index.min(), end=dedup.index.max(), freq="1h")
            reindexed = dedup.reindex(full_idx)
            reindexed["station"] = station_name

            # Forward fill static spatial coordinates
            for coord in ["latitude", "longitude"]:
                if coord in reindexed.columns:
                    reindexed[coord] = reindexed[coord].bfill().ffill()

            reindexed = reindexed.reset_index().rename(columns={"index": "timestamp"})
            regularized_stations.append(reindexed)

        full_regularized = pd.concat(regularized_stations, ignore_index=True)
        return full_regularized

    def impute_short_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply time-based linear interpolation strictly for short gaps (<= max_gap hours).
        Long gaps (> max_gap hours) are left as NaN to prevent fake synthetic stability.
        """
        df_imputed = df.copy()
        imputation_counts = {}

        cols_to_interpolate = [c for c in PHYSICAL_BOUNDS.keys() if c in df_imputed.columns]

        for col in cols_to_interpolate:
            n_missing_before = int(df_imputed[col].isna().sum())

            # Group by station so interpolation does not cross station boundaries
            df_imputed[col] = df_imputed.groupby("station")[col].transform(
                lambda s: s.interpolate(method="linear", limit=self.max_gap, limit_direction="forward")
            )

            n_missing_after = int(df_imputed[col].isna().sum())
            imputation_counts[col] = n_missing_before - n_missing_after

        return df_imputed, imputation_counts

    def clean_telemetry(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run full end-to-end sanitization pipeline.
        Returns cleaned DataFrame and a Data Quality Audit Report.
        """
        logger.info("Starting Data Sanitization Pipeline on %d raw rows...", len(df))

        # Step 1: Regularize time grid
        df_grid = self.regularize_time_grid(df)

        # Step 2: Validate physical instrument bounds
        df_bounded, bounds_report = self.validate_physical_bounds(df_grid)

        # Step 3: Detect stuck/frozen sensors
        df_unstuck, stuck_count = self.detect_stuck_sensors(df_bounded)

        # Step 4: Strict short-gap interpolation
        df_sanitized, interp_report = self.impute_short_gaps(df_unstuck)

        audit_report = {
            "initial_rows": len(df),
            "regularized_rows": len(df_sanitized),
            "bounds_violations": bounds_report,
            "stuck_sensor_values_purged": stuck_count,
            "short_gaps_imputed": interp_report,
            "remaining_pm25_nans": int(df_sanitized[PRIMARY_TARGET].isna().sum()),
        }

        logger.info("Data Sanitization complete: %d rows retained, %d short PM2.5 gaps imputed.",
                    len(df_sanitized), interp_report.get(PRIMARY_TARGET, 0))
        return df_sanitized, audit_report
