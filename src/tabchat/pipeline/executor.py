"""Deterministic TabPFN executor with data splitting, local baselines, and caching."""

from __future__ import annotations

import io
import logging
from typing import Any
import numpy as np
import pandas as pd

from tabchat.metrics import (
    MajorityClassBaseline,
    MeanTargetBaseline,
    compute_classification_metrics,
    compute_regression_metrics,
)
from tabchat.spec import (
    JobSpec,
    compute_dataset_hash,
    compute_spec_hash,
    validate_job_spec,
)
from tabchat.tabpfn import DiskFitCache, FakeTabPFNClient, TabPFNClient

logger = logging.getLogger("tabchat.executor")


class ExecutorError(Exception):
    """Raised when pipeline execution fails."""


def split_dataframe(
    df: pd.DataFrame,
    spec: JobSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically partition a DataFrame into (train_df, holdout_df).

    Args:
        df: Input DataFrame.
        spec: Validated JobSpec containing split strategy and fraction.

    Returns:
        (train_df, holdout_df)
    """
    n_rows = len(df)
    if n_rows < 2:
        raise ExecutorError(f"Cannot split dataset with only {n_rows} row.")

    # 1. Random Split
    if spec.split_strategy == "random":
        rng = np.random.RandomState(spec.random_seed)
        shuffled_indices = rng.permutation(n_rows)
        n_holdout = max(1, int(round(n_rows * spec.holdout_fraction)))
        n_train = n_rows - n_holdout
        if n_train < 1:
            n_train = 1
            n_holdout = n_rows - 1

        train_idx = shuffled_indices[:n_train]
        holdout_idx = shuffled_indices[n_train:]
        return df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy()

    # 2. Chronological Split
    elif spec.split_strategy == "chronological":
        if not spec.split_column or spec.split_column not in df.columns:
            raise ExecutorError(f"Split column '{spec.split_column}' not found for chronological split.")

        sorted_df = df.sort_values(by=spec.split_column, ascending=True)
        n_holdout = max(1, int(round(n_rows * spec.holdout_fraction)))
        n_train = n_rows - n_holdout
        if n_train < 1:
            n_train = 1
            n_holdout = n_rows - 1

        train_df = sorted_df.iloc[:n_train].copy()
        holdout_df = sorted_df.iloc[n_train:].copy()
        return train_df, holdout_df

    # 3. Group Split
    elif spec.split_strategy == "group":
        if not spec.split_column or spec.split_column not in df.columns:
            raise ExecutorError(f"Split column '{spec.split_column}' not found for group split.")

        unique_groups = df[spec.split_column].unique()
        if len(unique_groups) < 2:
            raise ExecutorError(
                f"Group split requires at least 2 distinct groups, but column '{spec.split_column}' has {len(unique_groups)}."
            )

        rng = np.random.RandomState(spec.random_seed)
        shuffled_groups = rng.permutation(unique_groups)

        target_holdout_rows = int(round(n_rows * spec.holdout_fraction))
        holdout_groups: set[Any] = set()
        accumulated_rows = 0

        for group in shuffled_groups:
            group_count = int((df[spec.split_column] == group).sum())
            holdout_groups.add(group)
            accumulated_rows += group_count
            if accumulated_rows >= target_holdout_rows and len(holdout_groups) < len(unique_groups):
                break

        # Fallback safeguard: ensure both partitions have >= 1 group
        if len(holdout_groups) == len(unique_groups):
            holdout_groups.remove(shuffled_groups[-1])

        train_mask = ~df[spec.split_column].isin(holdout_groups)
        holdout_mask = df[spec.split_column].isin(holdout_groups)

        return df[train_mask].copy(), df[holdout_mask].copy()

    else:
        raise ExecutorError(f"Unsupported split strategy '{spec.split_strategy}'.")


class TabPFNExecutor:
    """Zero-LLM deterministic executor running TabPFN, local baselines, and evaluation."""

    def __init__(
        self,
        tabpfn_client: TabPFNClient | FakeTabPFNClient | None = None,
        cache: DiskFitCache | None = None,
    ) -> None:
        self.client = tabpfn_client or FakeTabPFNClient()
        self.cache = cache or DiskFitCache()

    async def execute(
        self,
        raw_csv_bytes: bytes,
        dataset_card: dict[str, Any],
        spec: JobSpec,
    ) -> dict[str, Any]:
        """Execute deterministic pipeline for a given table and JobSpec."""
        # 1. Spec validation against dataset card
        spec_errors = validate_job_spec(spec, dataset_card)
        if spec_errors:
            raise ExecutorError(f"Invalid JobSpec: {'; '.join(spec_errors)}")

        # 2. Parse DataFrame
        df = pd.read_csv(io.BytesIO(raw_csv_bytes), encoding="utf-8-sig")

        # 3. Deterministic Train/Holdout Split
        train_df, holdout_df = split_dataframe(df, spec)
        y_train = train_df[spec.target_column].to_numpy()
        y_holdout = holdout_df[spec.target_column].to_numpy()

        # 4. Compute Local Baseline on Holdout Set (0 LLM Calls)
        if spec.task_type == "classification":
            baseline_clf = MajorityClassBaseline().fit(y_train)
            baseline_metrics = baseline_clf.evaluate(y_holdout)
        else:
            baseline_reg = MeanTargetBaseline().fit(y_train)
            baseline_metrics = baseline_reg.evaluate(y_holdout)

        # 5. Check Disk Fit Cache
        dataset_hash = compute_dataset_hash(raw_csv_bytes)
        spec_hash = compute_spec_hash(spec)
        cached_run = self.cache.get(dataset_hash, spec_hash)

        cache_hit = False
        if cached_run is not None:
            cache_hit = True
            raw_predictions = cached_run
        else:
            # 6. Invoke TabPFN (Real REST or Fake client)
            # Restrict columns strictly to feature_columns + target_column
            train_subset = train_df[spec.feature_columns + [spec.target_column]].copy()
            holdout_subset = holdout_df[spec.feature_columns + [spec.target_column]].copy()

            raw_predictions = await self.client.fit_and_predict(
                train_df=train_subset,
                test_df=holdout_subset,
                target_col=spec.target_column,
                task=spec.task_type,
            )
            # Store in cache
            self.cache.set(dataset_hash, spec_hash, raw_predictions)

        # 7. Compute Model Holdout Metrics
        sample_predictions: list[dict[str, Any]] = []

        if spec.task_type == "classification":
            probs = np.array(raw_predictions["probabilities"])
            preds = np.array(raw_predictions["predictions"])
            classes = raw_predictions.get("classes")

            model_metrics = compute_classification_metrics(
                y_true=y_holdout,
                y_prob=probs,
                y_pred=preds,
                classes=classes,
            )

            # Baseline comparison
            acc_delta = round(model_metrics.get("accuracy", 0.0) - baseline_metrics.get("accuracy", 0.0), 4)
            b_acc = baseline_metrics.get("accuracy", 0.0)
            acc_lift = round(model_metrics.get("accuracy", 0.0) / b_acc, 2) if b_acc > 0 else 1.0
            comparison = {
                "accuracy_delta": acc_delta,
                "accuracy_lift": acc_lift,
                "baseline_accuracy": baseline_metrics.get("accuracy"),
                "model_accuracy": model_metrics.get("accuracy"),
            }
            if "roc_auc" in model_metrics:
                comparison["roc_auc_delta"] = round(model_metrics["roc_auc"] - 0.5, 4)

            # Sample predictions
            for idx in range(min(5, len(holdout_df))):
                true_val = str(y_holdout[idx])
                pred_val = str(preds[idx])
                # Confidence is max predicted probability
                confidence = float(np.max(probs[idx])) if probs.ndim == 2 else float(probs[idx])
                sample_predictions.append({
                    "sample_index": int(idx),
                    "true_value": true_val,
                    "predicted_value": pred_val,
                    "confidence": round(confidence, 4),
                    "is_correct": bool(true_val == pred_val),
                })

        else:  # regression
            preds = np.array(raw_predictions["predictions"], dtype=float)
            quantiles = {
                k: np.array(v, dtype=float)
                for k, v in raw_predictions.get("quantiles", {}).items()
            } if spec.confidence_intervals else None

            model_metrics = compute_regression_metrics(
                y_true=y_holdout,
                y_pred=preds,
                quantiles=quantiles,
            )

            # Baseline comparison
            b_rmse = baseline_metrics.get("rmse", 0.0)
            m_rmse = model_metrics.get("rmse", 0.0)
            rmse_reduction = round(b_rmse - m_rmse, 4)
            rmse_pct = round((rmse_reduction / b_rmse) * 100.0, 2) if b_rmse > 0 else 0.0

            comparison = {
                "rmse_reduction": rmse_reduction,
                "rmse_reduction_percent": rmse_pct,
                "baseline_rmse": b_rmse,
                "model_rmse": m_rmse,
                "model_r2": model_metrics.get("r2"),
            }

            # Sample predictions
            for idx in range(min(5, len(holdout_df))):
                true_val = float(y_holdout[idx])
                pred_val = float(preds[idx])
                sample_item: dict[str, Any] = {
                    "sample_index": int(idx),
                    "true_value": round(true_val, 4),
                    "predicted_value": round(pred_val, 4),
                    "abs_error": round(abs(true_val - pred_val), 4),
                }
                if quantiles and "q10" in quantiles and "q90" in quantiles:
                    q10_val = float(quantiles["q10"][idx])
                    q90_val = float(quantiles["q90"][idx])
                    sample_item["interval_80"] = [round(q10_val, 4), round(q90_val, 4)]
                    sample_item["in_interval"] = bool(q10_val <= true_val <= q90_val)
                sample_predictions.append(sample_item)

        results: dict[str, Any] = {
            "task_type": spec.task_type,
            "target_column": spec.target_column,
            "feature_columns": spec.feature_columns,
            "split_strategy": spec.split_strategy,
            "split_column": spec.split_column,
            "n_train": len(train_df),
            "n_holdout": len(holdout_df),
            "baseline_metrics": baseline_metrics,
            "model_metrics": model_metrics,
            "comparison": comparison,
            "sample_predictions": sample_predictions,
            "cache_hit": cache_hit,
        }

        return results
