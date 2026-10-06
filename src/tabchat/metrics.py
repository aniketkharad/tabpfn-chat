"""Deterministic local baselines and pure-NumPy evaluation metrics for tabchat."""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute the two-sided Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Total number of trials.
        z: Z-critical value (default 1.96 for 95% confidence).

    Returns:
        (lower_bound, upper_bound) in range [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 1.0
    p_hat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = p_hat + (z * z) / (2.0 * float(n))
    spread = z * math.sqrt(
        (p_hat * (1.0 - p_hat)) / float(n) + (z * z) / (4.0 * float(n) * float(n))
    )
    lower = max(0.0, (center - spread) / denom)
    upper = min(1.0, (center + spread) / denom)
    return round(lower, 4), round(upper, 4)


def wilson_lower_bound(k: int, n: int, z: float = 1.645) -> float:
    """Compute the one-sided Wilson lower bound (default 90% confidence)."""
    lower, _ = wilson_interval(k, n, z=z)
    return lower


def compute_roc_auc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute binary ROC-AUC using the Mann-Whitney U rank statistic in pure NumPy.

    Args:
        y_true: 1D binary labels (0 or 1).
        y_prob: 1D predicted probabilities for positive class.

    Returns:
        ROC-AUC score between 0.0 and 1.0. Returns 0.5 if single class present.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank probabilities with average ties handling
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)

    # Average ranks for duplicate probabilities
    _, inverse_indices, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    tie_groups = np.where(counts > 1)[0]
    for group_idx in tie_groups:
        indices = np.where(inverse_indices == group_idx)[0]
        ranks[indices] = np.mean(ranks[indices])

    rank_sum_pos = np.sum(ranks[pos_mask])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(round(auc, 4))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and reliability bin table for binary classification.

    Args:
        y_true: 1D binary labels (0 or 1).
        y_prob: 1D predicted probabilities for class 1 in [0, 1].
        n_bins: Number of probability bins (default 10).

    Returns:
        (ece_score, reliability_bins_list)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)
    reliability_bins: list[dict[str, Any]] = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        bin_range = [round(float(bins[b]), 2), round(float(bins[b + 1]), 2)]

        if count == 0:
            reliability_bins.append({
                "bin": b,
                "range": bin_range,
                "count": 0,
                "mean_pred": None,
                "empirical_rate": None,
            })
            continue

        mean_pred = float(np.mean(y_prob[mask]))
        empirical_rate = float(np.mean(y_true[mask]))
        gap = abs(empirical_rate - mean_pred)
        ece += (count / total_samples) * gap

        reliability_bins.append({
            "bin": b,
            "range": bin_range,
            "count": count,
            "mean_pred": round(mean_pred, 4),
            "empirical_rate": round(empirical_rate, 4),
            "calibration_gap": round(gap, 4),
        })

    return float(round(ece, 4)), reliability_bins


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    classes: list[Any] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification performance and calibration metrics."""
    y_true = np.asarray(y_true)
    n_samples = len(y_true)

    if n_samples == 0:
        return {}

    # Extract distinct classes
    if classes is None:
        unique_classes = list(np.unique(y_true))
    else:
        unique_classes = list(classes)

    is_binary = len(unique_classes) <= 2

    # Derived predicted class labels if not provided
    if y_pred is None:
        if is_binary and (y_prob.ndim == 1 or y_prob.shape[-1] == 1):
            y_pred = (y_prob.squeeze() >= 0.5).astype(int)
            if len(unique_classes) == 2 and not np.array_equal(unique_classes, [0, 1]):
                y_pred = np.where(y_pred == 1, unique_classes[1], unique_classes[0])
        elif y_prob.ndim == 2:
            max_idx = np.argmax(y_prob, axis=1)
            y_pred = np.array([unique_classes[idx] for idx in max_idx])
        else:
            y_pred = np.asarray(y_prob)

    # 1. Accuracy & Wilson interval
    correct = int(np.sum(y_true == y_pred))
    accuracy = float(correct / n_samples)
    acc_lower, acc_upper = wilson_interval(correct, n_samples)

    # 2. Balanced Accuracy
    per_class_recalls = []
    for c in unique_classes:
        mask = y_true == c
        if np.sum(mask) > 0:
            recall = np.sum((y_pred == c) & mask) / np.sum(mask)
            per_class_recalls.append(recall)
    balanced_accuracy = float(np.mean(per_class_recalls)) if per_class_recalls else accuracy

    metrics: dict[str, Any] = {
        "accuracy": round(accuracy, 4),
        "accuracy_ci_95": [acc_lower, acc_upper],
        "balanced_accuracy": round(balanced_accuracy, 4),
        "n_samples": n_samples,
        "classes": [str(c) for c in unique_classes],
    }

    # 3. ROC-AUC, Log Loss, and ECE
    if is_binary:
        pos_class = unique_classes[-1]
        binary_y_true = (y_true == pos_class).astype(int)

        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob.squeeze()

        prob_clipped = np.clip(prob_pos, 1e-15, 1.0 - 1e-15)
        # Log loss
        log_loss_val = -float(
            np.mean(
                binary_y_true * np.log(prob_clipped)
                + (1 - binary_y_true) * np.log(1.0 - prob_clipped)
            )
        )
        roc_auc_val = compute_roc_auc_binary(binary_y_true, prob_pos)
        ece_val, reliability_bins = compute_ece(binary_y_true, prob_pos)

        metrics["roc_auc"] = roc_auc_val
        metrics["log_loss"] = round(log_loss_val, 4)
        metrics["ece"] = ece_val
        metrics["reliability_bins"] = reliability_bins
    elif y_prob.ndim == 2 and y_prob.shape[1] == len(unique_classes):
        # Multiclass OvR macro ROC-AUC
        ovr_aucs = []
        for c_idx, c_val in enumerate(unique_classes):
            bin_y = (y_true == c_val).astype(int)
            if np.sum(bin_y == 1) > 0 and np.sum(bin_y == 0) > 0:
                auc_c = compute_roc_auc_binary(bin_y, y_prob[:, c_idx])
                ovr_aucs.append(auc_c)
        if ovr_aucs:
            metrics["roc_auc"] = round(float(np.mean(ovr_aucs)), 4)

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive regression performance and interval calibration metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_samples = len(y_true)

    if n_samples == 0:
        return {}

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # R-squared
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics: dict[str, Any] = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_samples": n_samples,
    }

    # Prediction interval coverage and width if 10th and 90th percentiles provided
    if quantiles and "q10" in quantiles and "q90" in quantiles:
        q10 = np.asarray(quantiles["q10"], dtype=float)
        q90 = np.asarray(quantiles["q90"], dtype=float)
        in_interval = (y_true >= q10) & (y_true <= q90)
        coverage = float(np.mean(in_interval))
        mean_width = float(np.mean(q90 - q10))
        metrics["interval_coverage_80"] = round(coverage, 4)
        metrics["mean_interval_width"] = round(mean_width, 4)

    return metrics


class MajorityClassBaseline:
    """Deterministic majority-class baseline for classification."""

    def __init__(self) -> None:
        self.majority_class: Any = None
        self.empirical_distribution: dict[Any, float] = {}
        self.classes: list[Any] = []

    def fit(self, y_train: np.ndarray) -> "MajorityClassBaseline":
        y_train = np.asarray(y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        self.classes = list(unique)
        total = len(y_train)

        max_idx = np.argmax(counts)
        self.majority_class = unique[max_idx]
        self.empirical_distribution = {
            u: float(c / total) for u, c in zip(unique, counts)
        }
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.majority_class)

    def predict_proba(self, n_samples: int) -> np.ndarray:
        if len(self.classes) == 2:
            # Return positive class prior probability 1D
            pos_prob = self.empirical_distribution.get(self.classes[1], 0.5)
            return np.full(n_samples, pos_prob)
        # Multiclass probability matrix
        row = np.array([self.empirical_distribution.get(c, 0.0) for c in self.classes])
        return np.tile(row, (n_samples, 1))

    def evaluate(self, y_holdout: np.ndarray) -> dict[str, Any]:
        n = len(y_holdout)
        preds = self.predict(n)
        probs = self.predict_proba(n)
        metrics = compute_classification_metrics(
            y_holdout, probs, preds, classes=self.classes
        )
        metrics["roc_auc"] = 0.5  # Constant predictions have unranked AUC = 0.5
        return metrics


class MeanTargetBaseline:
    """Deterministic mean-target baseline for regression."""

    def __init__(self) -> None:
        self.mean_value: float = 0.0
        self.std_value: float = 0.0

    def fit(self, y_train: np.ndarray) -> "MeanTargetBaseline":
        y_arr = np.asarray(y_train, dtype=float)
        self.mean_value = float(np.mean(y_arr))
        self.std_value = float(np.std(y_arr))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.mean_value)

    def evaluate(self, y_holdout: np.ndarray) -> dict[str, Any]:
        preds = self.predict(len(y_holdout))
        # Compute baseline quantiles based on training standard deviation
        q10 = preds - 1.28 * self.std_value
        q90 = preds + 1.28 * self.std_value
        quantiles = {"q10": q10, "q90": q90}
        metrics = compute_regression_metrics(y_holdout, preds, quantiles=quantiles)
        return metrics
