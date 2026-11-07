"""Unit tests for evaluation metrics and loss calculations."""

import numpy as np
from src.models.demand_models import evaluate_demand_predictions, evaluate_interval_coverage
from src.models.expiry_models import evaluate_risk_predictions


def test_evaluate_demand_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = evaluate_demand_predictions(y_true, y_pred)
    # Errors: +10, -10, 0
    # MAE = 20 / 3 = 6.67
    # RMSE = sqrt(200 / 3) = 8.16
    # Total actual = 600, WAPE = (20 / 600) * 100 = 3.33%
    # Bias = (10 - 10 + 0) / 3 = 0.0

    assert np.isclose(metrics["mae"], 6.67, atol=0.01)
    assert np.isclose(metrics["rmse"], 8.16, atol=0.01)
    assert np.isclose(metrics["wape"], 3.33, atol=0.01)
    assert np.isclose(metrics["bias"], 0.0, atol=0.01)


def test_evaluate_interval_coverage():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    p10 = np.array([5.0, 15.0, 25.0, 50.0])   # 4th item out of bounds
    p90 = np.array([15.0, 25.0, 35.0, 60.0])

    cov = evaluate_interval_coverage(y_true, p10, p90)
    # 3 out of 4 in interval = 75.0%
    assert np.isclose(cov["coverage_pct"], 75.0)
    assert np.isclose(cov["mean_interval_width"], 10.0)


def test_evaluate_risk_metrics_and_cost():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])  # 1 FP, 1 FN, 2 TP, 1 TN

    metrics = evaluate_risk_predictions(y_true, y_pred, cost_fn=5.0, cost_fp=1.0)
    # TN=1, FP=1, FN=1, TP=2
    # Precision = 2 / (2 + 1) = 0.6667
    # Recall = 2 / (2 + 1) = 0.6667
    # Cost = 5.0 * 1 + 1.0 * 1 = 6.0
    assert np.isclose(metrics["precision"], 0.6667, atol=0.001)
    assert np.isclose(metrics["recall"], 0.6667, atol=0.001)
    assert metrics["business_cost"] == 6.0
