"""
Atmospheric Spatial Graph Engine for Multi-Station Telemetry.
Computes geographic distance matrices, compass bearing vectors, and dynamic
wind-directed plume advection weights to model cross-city smoke transport.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import PRIMARY_TARGET

logger = logging.getLogger(__name__)

# Known geographic coordinates for Delhi CAAQMS monitoring network
STATION_COORDINATES = {
    "Anand Vihar, Delhi - DPCC": (28.6476, 77.3158),
    "Punjabi Bagh, Delhi - DPCC": (28.6740, 77.1310),
    "R K Puram, Delhi - DPCC": (28.5632, 77.1869),
    "Bawana, Delhi - DPCC": (28.7762, 77.0511),
    "ITO, Delhi - CPCB": (28.6286, 77.2410),
}


class AtmosphericSpatialGraph:
    """Computes dynamic, wind-aligned spatial features across a network of monitoring stations."""

    EARTH_RADIUS_KM = 6371.0

    def __init__(self, coordinates: Dict[str, Tuple[float, float]] = None):
        self.coords = coordinates or STATION_COORDINATES
        self.stations = list(self.coords.keys())
        self.distance_matrix = self._compute_distance_matrix()
        self.bearing_matrix = self._compute_bearing_matrix()

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute great-circle distance between two GPS points in kilometers."""
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return self.EARTH_RADIUS_KM * c

    def _compass_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute compass bearing (in degrees, 0-360) pointing from point 1 to point 2."""
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dlambda = np.radians(lon2 - lon1)

        x = np.sin(dlambda) * np.cos(phi2)
        y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
        initial_bearing = np.degrees(np.arctan2(x, y))
        return (initial_bearing + 360.0) % 360.0

    def _compute_distance_matrix(self) -> pd.DataFrame:
        """Pairwise distance matrix (in km) between all stations."""
        matrix = pd.DataFrame(index=self.stations, columns=self.stations, dtype=float)
        for s1 in self.stations:
            lat1, lon1 = self.coords[s1]
            for s2 in self.stations:
                lat2, lon2 = self.coords[s2]
                matrix.loc[s1, s2] = self._haversine_distance(lat1, lon1, lat2, lon2)
        return matrix

    def _compute_bearing_matrix(self) -> pd.DataFrame:
        """Compass bearing matrix (in degrees) from station i pointing toward station j."""
        matrix = pd.DataFrame(index=self.stations, columns=self.stations, dtype=float)
        for s1 in self.stations:
            lat1, lon1 = self.coords[s1]
            for s2 in self.stations:
                if s1 == s2:
                    matrix.loc[s1, s2] = 0.0
                else:
                    lat2, lon2 = self.coords[s2]
                    matrix.loc[s1, s2] = self._compass_bearing(lat1, lon1, lat2, lon2)
        return matrix

    def compute_dynamic_upwind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dynamic wind-aligned spatial cross-station features for each observation.
        Adds:
        - `pm25_upwind_current`: Current weighted PM2.5 at upwind stations.
        - `pm25_upwind_lag3`: Upwind PM2.5 3 hours ago (accounting for travel time).
        - `upwind_advection_flux`: Wind speed * upwind PM2.5 burden.
        """
        df_out = df.copy()
        logger.info("Computing dynamic wind-aligned spatial graph features across %d rows...", len(df_out))

        # Subselect stations present in the dataframe
        active_stations = [s for s in self.stations if s in df_out["station"].unique()]
        dist_sub = self.distance_matrix.loc[active_stations, active_stations].values
        bearing_sub = self.bearing_matrix.loc[active_stations, active_stations].values

        # Pivot to create synchronized cross-station wide tables for PM2.5 and PM2.5 lag_3
        wide_pm25 = df_out.pivot(index="timestamp", columns="station", values=PRIMARY_TARGET)[active_stations]
        wide_pm25_lag3 = wide_pm25.shift(3)

        upwind_current_vals = np.zeros(len(df_out))
        upwind_lag3_vals = np.zeros(len(df_out))
        advection_flux_vals = np.zeros(len(df_out))

        timestamps = df_out["timestamp"].values
        stations = df_out["station"].values
        wind_dirs = df_out["wind_direction"].values
        wind_speeds = df_out["wind_speed"].values

        # Meteorological convention: Wind direction is where wind comes FROM.
        # Plume propagation direction = (wind_direction + 180) % 360
        transport_dirs = (wind_dirs + 180.0) % 360.0

        time_to_idx = {t: i for i, t in enumerate(wide_pm25.index.values)}
        wide_pm25_arr = wide_pm25.values
        wide_lag3_arr = wide_pm25_lag3.values
        station_to_col = {s: i for i, s in enumerate(active_stations)}

        # Replace 0 on diagonal with inf for distance decay
        inv_dist = 1.0 / np.where(dist_sub == 0, np.inf, dist_sub)

        for row_idx in range(len(df_out)):
            t_val = timestamps[row_idx]
            target_station = stations[row_idx]
            t_idx = time_to_idx.get(t_val)

            if t_idx is None or target_station not in station_to_col:
                continue

            j_col = station_to_col[target_station]
            t_dir = transport_dirs[row_idx]
            ws = wind_speeds[row_idx]

            # Bearing from all other stations i pointing toward target station j
            bearings_to_j = bearing_sub[:, j_col]
            distances_to_j = inv_dist[:, j_col]

            # Angle difference between plume propagation direction and vector i -> j
            angle_diff = np.radians(t_dir - bearings_to_j)
            alignment = np.maximum(0.0, np.cos(angle_diff))

            # Weight of station i impacting target station j
            # Station cannot impact itself (alignment[j] set to 0)
            alignment[j_col] = 0.0

            weights = alignment * distances_to_j
            weight_sum = np.sum(weights)

            if weight_sum > 1e-6:
                norm_weights = weights / weight_sum
                # Current upwind
                pm_row = wide_pm25_arr[t_idx, :]
                lag3_row = wide_lag3_arr[t_idx, :]

                # Handle any NaNs in upwind stations cleanly
                valid_mask = ~np.isnan(pm_row)
                if np.any(valid_mask & (norm_weights > 0)):
                    adj_weights = norm_weights * valid_mask
                    adj_sum = np.sum(adj_weights)
                    if adj_sum > 0:
                        adj_weights = adj_weights / adj_sum
                        upwind_current_vals[row_idx] = np.sum(adj_weights * np.nan_to_num(pm_row))

                valid_lag3 = ~np.isnan(lag3_row)
                if np.any(valid_lag3 & (norm_weights > 0)):
                    adj_lag_weights = norm_weights * valid_lag3
                    adj_lag_sum = np.sum(adj_lag_weights)
                    if adj_lag_sum > 0:
                        adj_lag_weights = adj_lag_weights / adj_lag_sum
                        upwind_lag3_vals[row_idx] = np.sum(adj_lag_weights * np.nan_to_num(lag3_row))

                advection_flux_vals[row_idx] = ws * upwind_current_vals[row_idx]
            else:
                # Calm air or no upwind stations: default to station's own concentration
                pm_self = wide_pm25_arr[t_idx, j_col]
                upwind_current_vals[row_idx] = pm_self if not np.isnan(pm_self) else 0.0
                lag3_self = wide_lag3_arr[t_idx, j_col]
                upwind_lag3_vals[row_idx] = lag3_self if not np.isnan(lag3_self) else 0.0
                advection_flux_vals[row_idx] = ws * upwind_current_vals[row_idx]

        df_out["pm25_upwind_current"] = np.round(upwind_current_vals, 2)
        df_out["pm25_upwind_lag3"] = np.round(upwind_lag3_vals, 2)
        df_out["upwind_advection_flux"] = np.round(advection_flux_vals, 2)

        return df_out
