---
name: uncertainty_and_metrics
description: Interpreting holdout accuracy vs. majority-class baseline, R2 vs. mean baseline, explaining probability calibration.
tags: [uncertainty, baselines, calibration, metrics]
---

# Uncertainty, Baselines, and Evaluation Metrics Recipe

## 1. Comparing Performance Against Local Trivial Baselines
- **Classification (Majority-Class Baseline)**:
  - Raw holdout accuracy must never be evaluated in isolation. On an imbalanced dataset where 90% of samples belong to Class A, predicting Class A for all rows yields 90% accuracy with zero predictive intelligence.
  - Compute the Delta: `model_accuracy - majority_baseline_accuracy`. Meaningful models must deliver positive lift over the majority baseline.
  - Assess `log_loss` and `roc_auc` alongside accuracy to account for probability confidence and threshold invariance.
- **Regression (Mean-Target Baseline)**:
  - Predict the training set mean target value for all holdout rows.
  - The mean baseline has an $R^2$ of 0.0. A positive model $R^2$ demonstrates genuine explained variance, whereas negative $R^2$ indicates performance worse than simply predicting the historical mean.
  - Evaluate RMSE delta: `model_rmse - baseline_rmse` (a negative delta represents an error reduction).

## 2. Calibrated Probabilities and Expected Calibration Error (ECE)
- TabPFN generates posterior predictive probabilities derived from in-context Bayesian inference.
- Well-calibrated probabilities mean that predictions with 80% confidence are correct approximately 80% of the time.
- Expected Calibration Error (ECE) measures the absolute discrepancy between predicted confidences and empirical outcomes across probability bins.

## 3. Calibrated Prediction Intervals in Regression
- TabPFN Regressor generates full predictive distributions, yielding empirical quantiles ($q_{10}, q_{50}, q_{90}$).
- The 80% prediction interval is $[q_{10}, q_{90}]$. On calibrated regressions, approximately 80% of actual holdout values fall within this interval.
