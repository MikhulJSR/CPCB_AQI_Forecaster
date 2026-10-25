"""
Unit tests for Atmospheric Spatial Graph Engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.spatial_graph import AtmosphericSpatialGraph, STATION_COORDINATES


def test_haversine_distance_symmetry():
    """Verify that distance matrix is symmetric and diagonal is zero."""
    graph = AtmosphericSpatialGraph()
    dist_df = graph.distance_matrix

    for s1 in graph.stations:
        assert dist_df.loc[s1, s1] == 0.0
        for s2 in graph.stations:
            assert np.isclose(dist_df.loc[s1, s2], dist_df.loc[s2, s1], atol=1e-3)
            # Distance between any two different Delhi stations should be 5 - 45 km
            if s1 != s2:
                assert 5.0 < dist_df.loc[s1, s2] < 50.0


def test_compass_bearing_reciprocity():
    """Verify that bearing from A->B is approximately 180 degrees opposite of B->A."""
    graph = AtmosphericSpatialGraph()
    bearing_df = graph.bearing_matrix

    s1 = "Bawana, Delhi - DPCC"
    s2 = "Anand Vihar, Delhi - DPCC"

    b_12 = bearing_df.loc[s1, s2]  # Bawana to Anand Vihar (SE: ~120-130 deg)
    b_21 = bearing_df.loc[s2, s1]  # Anand Vihar to Bawana (NW: ~300-310 deg)

    diff = abs(b_12 - b_21)
    assert np.isclose(diff, 180.0, atol=2.0)


def test_wind_alignment_directionality():
    """
    Verify that when wind blows from NW (315 deg, plume travels SE at 135 deg):
    Anand Vihar (SE) receives high upwind weight from Bawana (NW),
    while Bawana receives 0 weight from Anand Vihar.
    """
    graph = AtmosphericSpatialGraph()

    # Create dummy data with 2 stations, wind blowing from 315 degrees (NW)
    times = pd.date_range("2025-10-01 00:00:00", periods=5, freq="1h")
    records = []
    for t in times:
        records.append({
            "timestamp": t,
            "station": "Bawana, Delhi - DPCC",
            "pm25": 400.0,
            "wind_speed": 4.0,
            "wind_direction": 315.0,  # From NW -> blowing towards SE
        })
        records.append({
            "timestamp": t,
            "station": "Anand Vihar, Delhi - DPCC",
            "pm25": 100.0,
            "wind_speed": 4.0,
            "wind_direction": 315.0,
        })
    df = pd.DataFrame(records)

    df_out = graph.compute_dynamic_upwind_features(df)

    # For Anand Vihar (downwind), upwind PM2.5 should reflect Bawana (400.0)
    av_rows = df_out[df_out["station"] == "Anand Vihar, Delhi - DPCC"]
    assert np.isclose(av_rows.iloc[0]["pm25_upwind_current"], 400.0, atol=5.0)

    # For Bawana (upwind source), Anand Vihar is DOWNWIND so cannot blow smoke back
    bawana_rows = df_out[df_out["station"] == "Bawana, Delhi - DPCC"]
    # When no upwind station exists, defaults to own concentration (400.0)
    assert bawana_rows.iloc[0]["pm25_upwind_current"] == 400.0
