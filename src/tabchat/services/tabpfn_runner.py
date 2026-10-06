"""Cached TabPFN service runner with local baselines, metric scoring, and atomic disk caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd

from tabchat.config import get_settings
from tabchat.schemas.job_spec import JobSpec

logger = logging.getLogger("tabchat.tabpfn_runner")

# Optional scikit-learn import
try:
    from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _json_serializer(obj: Any) -> Any:
    """Helper serializer for NumPy types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def compute_cache_key(raw_bytes: bytes, spec: JobSpec) -> str:
    """Compute deterministic SHA-256 cache key from raw CSV bytes and canonical JobSpec."""
    csv_hash = hashlib.sha256(raw_bytes).hexdigest()
    spec_json = spec.canonical_json()
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    return f"{csv_hash}_{spec_hash}"


def _split_data(
    df: pd.DataFrame,
    spec: JobSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition DataFrame into X_train, X_holdout, y_train, y_holdout based on spec."""
    X = df[spec.feature_columns].copy()
    y = df[spec.target_column].copy()
    n_rows = len(df)
    rng = np.random.RandomState(spec.random_seed)

    if spec.split_strategy == "random":
        if spec.task_type == "classification":
            # Stratified random split
            train_indices: list[int] = []
            holdout_indices: list[int] = []
            classes = y.unique()

            for c in classes:
                c_indices = df[y == c].index.to_numpy()
                shuffled_c = rng.permutation(c_indices)
                n_c = len(shuffled_c)
                n_c_holdout = max(1, int(round(n_c * spec.holdout_fraction)))
                # Ensure at least 1 in train if possible
                if n_c_holdout >= n_c and n_c > 1:
                    n_c_holdout = n_c - 1

                holdout_indices.extend(shuffled_c[:n_c_holdout])
                train_indices.extend(shuffled_c[n_c_holdout:])

            train_idx = np.array(train_indices)
            holdout_idx = np.array(holdout_indices)
        else:
            # Standard random permutation for regression
            shuffled = rng.permutation(n_rows)
            n_holdout = max(1, int(round(n_rows * spec.holdout_fraction)))
            if n_holdout >= n_rows:
                n_holdout = n_rows - 1

            train_idx = df.index[shuffled[n_holdout:]].to_numpy()
            holdout_idx = df.index[shuffled[:n_holdout]].to_numpy()

    elif spec.split_strategy == "chronological":
        if not spec.split_column or spec.split_column not in df.columns:
            raise ValueError(f"Split column '{spec.split_column}' not found for chronological split.")
        sorted_df = df.sort_values(by=spec.split_column, ascending=True)
        n_holdout = max(1, int(round(n_rows * spec.holdout_fraction)))
        if n_holdout >= n_rows:
            n_holdout = n_rows - 1

        train_idx = sorted_df.index[:-n_holdout].to_numpy()
        holdout_idx = sorted_df.index[-n_holdout:].to_numpy()

    elif spec.split_strategy == "group":
        if not spec.split_column or spec.split_column not in df.columns:
            raise ValueError(f"Split column '{spec.split_column}' not found for group split.")
        groups = df[spec.split_column].unique()
        if len(groups) < 2:
            raise ValueError(f"Group split requires at least 2 distinct groups in '{spec.split_column}'.")

        shuffled_groups = rng.permutation(groups)
        target_holdout_rows = int(round(n_rows * spec.holdout_fraction))

        holdout_groups: set[Any] = set()
        accumulated_rows = 0
        for grp in shuffled_groups:
            count = int((df[spec.split_column] == grp).sum())
            holdout_groups.add(grp)
            accumulated_rows += count
            if accumulated_rows >= target_holdout_rows and len(holdout_groups) < len(groups):
                break

        holdout_mask = df[spec.split_column].isin(holdout_groups)
        train_idx = df.index[~holdout_mask].to_numpy()
        holdout_idx = df.index[holdout_mask].to_numpy()
    else:
        raise ValueError(f"Unknown split strategy: {spec.split_strategy}")

    return X.loc[train_idx], X.loc[holdout_idx], y.loc[train_idx], y.loc[holdout_idx]


def _compute_baselines(
    task_type: str,
    y_train: pd.Series,
    y_holdout: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute local trivial baseline metrics (Majority class or Mean target)."""
    if task_type == "classification":
        majority_class = y_train.mode()[0]
        baseline_preds = np.full(len(y_holdout), majority_class)

        # Baseline accuracy
        if SKLEARN_AVAILABLE:
            base_acc = float(accuracy_score(y_holdout, baseline_preds))
        else:
            base_acc = float(np.mean(y_holdout.to_numpy() == majority_class))

        # Baseline log loss using empirical training priors
        unique_classes = sorted(y_train.unique())
        priors = np.array([(y_train == c).mean() for c in unique_classes], dtype=float)
        priors_matrix = np.tile(priors, (len(y_holdout), 1))

        if SKLEARN_AVAILABLE and len(unique_classes) >= 2:
            base_log_loss = float(log_loss(y_holdout, priors_matrix, labels=unique_classes))
        else:
            # Pure NumPy multiclass cross-entropy
            eps = 1e-15
            clipped_priors = np.clip(priors_matrix, eps, 1.0 - eps)
            one_hot = np.zeros_like(priors_matrix)
            for idx, val in enumerate(y_holdout):
                if val in unique_classes:
                    one_hot[idx, unique_classes.index(val)] = 1.0
            base_log_loss = -float(np.mean(np.sum(one_hot * np.log(clipped_priors), axis=1)))

        baseline_metrics = {
            "accuracy": round(base_acc, 4),
            "log_loss": round(base_log_loss, 4),
        }
        return baseline_metrics, baseline_preds

    else:  # regression
        train_mean = float(y_train.mean())
        baseline_preds = np.full(len(y_holdout), train_mean, dtype=float)

        y_holdout_arr = y_holdout.to_numpy(dtype=float)
        errors = y_holdout_arr - train_mean
        base_rmse = float(np.sqrt(np.mean(errors ** 2)))

        if SKLEARN_AVAILABLE:
            base_r2 = float(r2_score(y_holdout_arr, baseline_preds))
        else:
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((y_holdout_arr - float(np.mean(y_holdout_arr))) ** 2))
            base_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        baseline_metrics = {
            "rmse": round(base_rmse, 4),
            "r2": round(base_r2, 4),
        }
        return baseline_metrics, baseline_preds


def _fit_and_predict_tabpfn(
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute TabPFN model fitting and prediction via hosted API.

    Can be mocked in tests or will import tabpfn_client when available.
    """
    settings = get_settings()
    if settings.TABPFN_TOKEN:
        if "TABPFN_TOKEN" not in os.environ:
            os.environ["TABPFN_TOKEN"] = settings.TABPFN_TOKEN
        try:
            import tabpfn_client
            tabpfn_client.set_access_token(settings.TABPFN_TOKEN)
        except Exception:
            pass

    if task_type == "classification":
        try:
            from tabpfn_client import TabPFNClassifier
            clf = TabPFNClassifier(random_state=random_seed)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_holdout)
            probas = clf.predict_proba(X_holdout)
            row_sums = np.sum(probas, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probas = probas / row_sums
            classes = list(getattr(clf, "classes_", sorted(y_train.unique())))
            return {
                "predictions": preds,
                "probabilities": probas,
                "classes": classes,
            }
        except ImportError:
            # Fallback mock distribution if tabpfn-client is not installed
            classes = list(sorted(y_train.unique()))
            majority = y_train.mode()[0]
            preds = np.full(len(X_holdout), majority)
            probas = np.full((len(X_holdout), len(classes)), 1.0 / len(classes))
            return {
                "predictions": preds,
                "probabilities": probas,
                "classes": classes,
            }
    else:
        try:
            from tabpfn_client import TabPFNRegressor
            reg = TabPFNRegressor(random_state=random_seed)
            reg.fit(X_train, y_train)
            preds = reg.predict(X_holdout)
            try:
                raw_q = reg.predict(X_holdout, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
                q_arr = np.asarray(raw_q)
                if q_arr.ndim == 2 and q_arr.shape[1] == 3:
                    q10, q50, q90 = q_arr[:, 0], q_arr[:, 1], q_arr[:, 2]
                elif q_arr.ndim == 2 and q_arr.shape[0] == 3:
                    q10, q50, q90 = q_arr[0, :], q_arr[1, :], q_arr[2, :]
                else:
                    std_val = float(y_train.std()) if len(y_train) > 1 else 1.0
                    q10 = preds - 1.28 * std_val
                    q50 = preds
                    q90 = preds + 1.28 * std_val
            except Exception:
                std_val = float(y_train.std()) if len(y_train) > 1 else 1.0
                q10 = preds - 1.28 * std_val
                q50 = preds
                q90 = preds + 1.28 * std_val

            return {
                "predictions": preds,
                "quantiles": {
                    "q10": q10,
                    "q50": q50,
                    "q90": q90,
                },
            }
        except ImportError:
            # Fallback mock distribution if tabpfn-client is not installed
            mean_val = float(y_train.mean())
            std_val = float(y_train.std()) if len(y_train) > 1 else 1.0
            preds = np.full(len(X_holdout), mean_val)
            return {
                "predictions": preds,
                "quantiles": {
                    "q10": preds - 1.28 * std_val,
                    "q50": preds,
                    "q90": preds + 1.28 * std_val,
                },
            }


def _score_model(
    task_type: str,
    y_holdout: pd.Series,
    api_result: dict[str, Any],
) -> dict[str, float]:
    """Score model predictions against holdout target."""
    model_preds = np.asarray(api_result["predictions"])

    if task_type == "classification":
        if SKLEARN_AVAILABLE:
            acc = float(accuracy_score(y_holdout, model_preds))
        else:
            acc = float(np.mean(y_holdout.to_numpy() == model_preds))

        probas = np.asarray(api_result.get("probabilities", []))
        classes = api_result.get("classes", list(sorted(y_holdout.unique())))

        if len(probas) > 0 and len(classes) >= 2:
            if SKLEARN_AVAILABLE:
                loss = float(log_loss(y_holdout, probas, labels=classes))
            else:
                eps = 1e-15
                clipped = np.clip(probas, eps, 1.0 - eps)
                one_hot = np.zeros_like(probas)
                for idx, val in enumerate(y_holdout):
                    if val in classes:
                        one_hot[idx, classes.index(val)] = 1.0
                loss = -float(np.mean(np.sum(one_hot * np.log(clipped), axis=1)))
        else:
            loss = 0.0

        return {
            "accuracy": round(acc, 4),
            "log_loss": round(loss, 4),
        }

    else:  # regression
        y_holdout_arr = y_holdout.to_numpy(dtype=float)
        errors = y_holdout_arr - model_preds.astype(float)
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        if SKLEARN_AVAILABLE:
            r2 = float(r2_score(y_holdout_arr, model_preds.astype(float)))
        else:
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((y_holdout_arr - float(np.mean(y_holdout_arr))) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }


async def run_tabpfn_job(
    spec: JobSpec,
    df: pd.DataFrame,
    raw_bytes: bytes,
    cache_dir: Path | None = None,
    api_caller: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute TabPFN model pipeline with deterministic disk caching and baseline comparison."""
    settings = get_settings()
    c_dir = cache_dir or (settings.DATA_DIR / "cache" / "tabpfn")
    c_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check cache
    cache_key = compute_cache_key(raw_bytes, spec)
    cache_path = c_dir / f"{cache_key}.json"

    if cache_path.exists():
        logger.info("TabPFN cache hit: %s", cache_path.name)
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached_data
        except Exception as e:
            logger.warning("Cache file corrupted (%s), recalculating: %s", cache_path.name, e)
            cache_path.unlink(missing_ok=True)

    # Step 2: Prepare data partitions
    X_train, X_holdout, y_train, y_holdout = _split_data(df, spec)

    # Step 3: Compute Baseline Metrics locally
    baseline_metrics, _ = _compute_baselines(spec.task_type, y_train, y_holdout)

    # Step 4: Call TabPFN API via asyncio.to_thread with timeout
    timeout = float(settings.TABPFN_FIT_TIMEOUT_SECONDS)
    caller = api_caller or _fit_and_predict_tabpfn

    try:
        api_result = await asyncio.wait_for(
            asyncio.to_thread(
                caller,
                spec.task_type,
                X_train,
                y_train,
                X_holdout,
                spec.random_seed,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"TabPFN API execution timed out after {timeout} seconds.")

    # Step 5: Score Holdout Set and compute Delta
    model_metrics = _score_model(spec.task_type, y_holdout, api_result)

    delta: dict[str, float] = {}
    for metric_name, model_val in model_metrics.items():
        if metric_name in baseline_metrics:
            delta[metric_name] = round(model_val - baseline_metrics[metric_name], 4)

    # Step 6: Generate Predictions on Holdout (limit preview to 10 rows)
    model_preds = np.asarray(api_result["predictions"])
    preview_count = min(10, len(X_holdout))
    sample_predictions: list[dict[str, Any]] = []

    for i in range(preview_count):
        actual_val = y_holdout.iloc[i]
        pred_val = model_preds[i]

        sample_item: dict[str, Any] = {
            "row_index": int(X_holdout.index[i]),
            "actual": float(actual_val) if spec.task_type == "regression" else str(actual_val),
            "predicted": float(pred_val) if spec.task_type == "regression" else str(pred_val),
            "features": {k: _json_serializer(v) for k, v in X_holdout.iloc[i].to_dict().items()},
        }

        if spec.task_type == "regression" and "quantiles" in api_result:
            q_dict = api_result["quantiles"]
            sample_item["quantiles"] = {
                "q10": round(float(q_dict["q10"][i]), 4),
                "q50": round(float(q_dict["q50"][i]), 4),
                "q90": round(float(q_dict["q90"][i]), 4),
            }
        elif spec.task_type == "classification" and "probabilities" in api_result:
            sample_item["probabilities"] = [
                round(float(p), 4) for p in api_result["probabilities"][i]
            ]

        sample_predictions.append(sample_item)

    # Step 7: Serialize full execution result dict and atomically write to cache
    warnings: list[str] = []
    if spec.task_type == "classification":
        class_counts = df[spec.target_column].value_counts()
        min_count = int(class_counts.min())
        total_rows = len(df)
        if min_count < 5 or (min_count / total_rows) < 0.05:
            warnings.append(
                f"Severe class imbalance detected: minority class has only {min_count} samples ({min_count / total_rows:.1%}). "
                "Holdout metrics have low statistical support and high variance."
            )
        if delta.get("accuracy", 0.0) <= 0.0:
            warnings.append("Model accuracy did not achieve positive lift over majority class baseline.")
    if spec.task_type == "regression" and model_metrics.get("r2", 0.0) <= 0.0:
        warnings.append("Model R2 is <= 0.0, indicating predictions do not outperform mean baseline.")

    if len(spec.feature_columns) >= 2:
        feats_df = df[spec.feature_columns]
        identical_pairs = []
        for i in range(len(spec.feature_columns)):
            for j in range(i + 1, len(spec.feature_columns)):
                col_i = spec.feature_columns[i]
                col_j = spec.feature_columns[j]
                if feats_df[col_i].equals(feats_df[col_j]):
                    identical_pairs.append((col_i, col_j))
        if identical_pairs:
            pair_strs = [f"('{a}', '{b}')" for a, b in identical_pairs[:3]]
            warnings.append(
                f"Multi-collinear identical columns detected in feature set: {', '.join(pair_strs)}. "
                "Redundant duplicate features may introduce noise."
            )

    result: dict[str, Any] = {
        "spec": spec.model_dump(),
        "holdout_size": len(X_holdout),
        "train_size": len(X_train),
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "delta": delta,
        "sample_predictions": sample_predictions,
        "warnings": warnings,
    }

    # Atomic write to cache file: write to .tmp then rename
    tmp_path = cache_path.with_suffix(".tmp")
    serialized = json.dumps(result, indent=2, ensure_ascii=False, default=_json_serializer)
    tmp_path.write_text(serialized, encoding="utf-8")
    os.replace(tmp_path, cache_path)
    logger.info("TabPFN Cache Stored: %s", cache_path.name)

    return result


def run_tabpfn_job_sync(
    spec: JobSpec,
    df: pd.DataFrame,
    raw_bytes: bytes,
    cache_dir: Path | None = None,
    api_caller: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Synchronous convenience wrapper for run_tabpfn_job."""
    return asyncio.run(run_tabpfn_job(spec, df, raw_bytes, cache_dir=cache_dir, api_caller=api_caller))
