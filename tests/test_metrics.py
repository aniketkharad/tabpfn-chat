"""Unit tests for pure-NumPy evaluation metrics, calibration, and local baselines."""

import numpy as np
from tabchat.metrics import (
    MajorityClassBaseline,
    MeanTargetBaseline,
    compute_classification_metrics,
    compute_ece,
    compute_regression_metrics,
    compute_roc_auc_binary,
    wilson_interval,
    wilson_lower_bound,
)


def test_wilson_interval_bounds():
    """Verify Wilson interval properties at boundaries and normal values."""
    # 0 successes out of 100 trials: lower bound is 0.0, upper bound > 0.0
    lower_0, upper_0 = wilson_interval(0, 100)
    assert lower_0 == 0.0
    assert 0.0 < upper_0 < 0.05

    # 100 successes out of 100 trials: upper bound is 1.0, lower bound < 1.0
    lower_100, upper_100 = wilson_interval(100, 100)
    assert 0.95 < lower_100 < 1.0
    assert upper_100 == 1.0

    # Intermediate value (50/100): interval centered around 0.5
    low_50, up_50 = wilson_interval(50, 100, z=1.96)
    assert 0.39 < low_50 < 0.45
    assert 0.55 < up_50 < 0.61

    # One-sided lower bound
    lb = wilson_lower_bound(30, 100, z=1.645)
    assert 0.20 < lb < 0.30


def test_compute_roc_auc_binary():
    """Verify binary ROC-AUC rank calculation under various conditions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])

    # Perfect prediction
    y_prob_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_roc_auc_binary(y_true, y_prob_perfect) == 1.0

    # Inverted prediction
    y_prob_inverted = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert compute_roc_auc_binary(y_true, y_prob_inverted) == 0.0

    # Random / constant prediction
    y_prob_constant = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert compute_roc_auc_binary(y_true, y_prob_constant) == 0.5

    # With ties
    y_prob_ties = np.array([0.2, 0.5, 0.5, 0.5, 0.8, 0.9])
    auc = compute_roc_auc_binary(y_true, y_prob_ties)
    assert 0.5 < auc < 1.0


def test_compute_ece():
    """Verify Expected Calibration Error and reliability bin computation."""
    # Perfectly calibrated probabilities
    y_prob = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.array([1 if i < 5 else 0 for i in range(50)] + [1 if i < 45 else 0 for i in range(50)])

    ece, bins = compute_ece(y_true, y_prob, n_bins=10)
    assert isinstance(ece, float)
    assert ece >= 0.0
    assert len(bins) == 10
    assert any(b["count"] > 0 for b in bins)


def test_classification_metrics_summary():
    """Verify full classification metrics bundle."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.4, 0.85, 0.65, 0.15])

    metrics = compute_classification_metrics(y_true, y_prob)
    assert "accuracy" in metrics
    assert "accuracy_ci_95" in metrics
    assert "balanced_accuracy" in metrics
    assert "roc_auc" in metrics
    assert "log_loss" in metrics
    assert "ece" in metrics
    assert metrics["accuracy"] == 1.0  # Perfect thresholding at 0.5


def test_regression_metrics_summary():
    """Verify regression metrics bundle with prediction intervals."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 32.0, 38.0])
    quantiles = {
        "q10": np.array([8.0, 16.0, 28.0, 35.0]),
        "q90": np.array([14.0, 23.0, 35.0, 42.0]),
    }

    metrics = compute_regression_metrics(y_true, y_pred, quantiles=quantiles)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.9
    assert metrics["interval_coverage_80"] == 1.0  # All points inside [q10, q90]
    assert "mean_interval_width" in metrics


def test_majority_class_baseline():
    """Verify MajorityClassBaseline fitting and evaluation."""
    y_train = np.array(["cat", "dog", "dog", "dog", "bird"])
    y_holdout = np.array(["dog", "cat", "dog", "dog"])

    baseline = MajorityClassBaseline().fit(y_train)
    assert baseline.majority_class == "dog"

    metrics = baseline.evaluate(y_holdout)
    assert metrics["accuracy"] == 0.75  # 3 dogs out of 4 samples
    assert metrics["roc_auc"] == 0.5   # Constant baseline has unranked AUC = 0.5


def test_mean_target_baseline():
    """Verify MeanTargetBaseline fitting and evaluation."""
    y_train = np.array([10.0, 20.0, 30.0])  # mean = 20.0
    y_holdout = np.array([18.0, 22.0, 20.0])

    baseline = MeanTargetBaseline().fit(y_train)
    assert baseline.mean_value == 20.0

    metrics = baseline.evaluate(y_holdout)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["mae"] == round(float((2.0 + 2.0 + 0.0) / 3.0), 4)
