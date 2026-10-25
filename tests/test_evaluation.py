"""
Unit tests for Model Evaluation and GRAP Decision Metrics.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation.evaluator import ModelEvaluator


def test_continuous_metrics_computation():
    """Verify standard regression metric calculation including MASE."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    y_diurnal_baseline = np.array([120.0, 180.0, 320.0])

    evaluator = ModelEvaluator()
    metrics = evaluator.compute_continuous_metrics(y_true, y_pred, y_diurnal_baseline)

    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
    assert metrics["r2"] > 0.95
    assert metrics["diurnal_baseline_mae"] == 20.0
    # MASE = 10 / 20 = 0.50
    assert np.isclose(metrics["mase"], 0.50, atol=1e-2)


def test_grap_exceedance_metrics():
    """Verify regulatory decision metrics (CSI, Precision, Recall)."""
    # Threshold = 250.0
    y_true = np.array([240.0, 260.0, 280.0, 220.0, 290.0])
    y_pred = np.array([230.0, 270.0, 240.0, 210.0, 310.0])
    # Truth binary:  [0, 1, 1, 0, 1] (3 positives)
    # Pred binary:   [0, 1, 0, 0, 1] (2 hits, 1 miss, 0 false alarms)

    evaluator = ModelEvaluator(grap_threshold=250.0)
    grap_metrics = evaluator.compute_grap_exceedance_metrics(y_true, y_pred)

    assert grap_metrics["true_positives_hits"] == 2
    assert grap_metrics["false_negatives_misses"] == 1
    assert grap_metrics["false_positives_false_alarms"] == 0
    assert grap_metrics["true_negatives"] == 2

    # Precision = 2 / (2 + 0) = 1.0
    assert grap_metrics["precision"] == 1.0
    # Recall = 2 / (2 + 1) = 0.6667
    assert np.isclose(grap_metrics["recall"], 0.6667, atol=1e-3)
    # CSI = 2 / (2 + 0 + 1) = 0.6667
    assert np.isclose(grap_metrics["csi_threat_score"], 0.6667, atol=1e-3)


def test_probabilistic_metrics_computation():
    """Verify pinball loss and PICP interval coverage calculation."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    q10 = np.array([80.0, 170.0, 260.0, 350.0, 450.0])
    q50 = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    q90 = np.array([120.0, 230.0, 340.0, 450.0, 550.0])

    evaluator = ModelEvaluator()
    metrics = evaluator.compute_probabilistic_metrics(y_true, q10, q50, q90)

    # All true points are within [q10, q90] -> coverage = 100%
    assert metrics["picp_coverage_pct"] == 100.0
    # Average width = (40 + 60 + 80 + 100 + 100) / 5 = 76.0
    assert metrics["mpiw_interval_width"] == 76.0
    # When q50 matches y_true, median pinball loss is 0.0
    assert metrics["pinball_loss_p50"] == 0.0
