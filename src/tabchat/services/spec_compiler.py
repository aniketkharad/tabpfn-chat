"""Spec compiler and dataset verifier for tabchat."""

from __future__ import annotations

import pandas as pd
from tabchat.schemas.job_spec import JobSpec


def compile_and_verify_spec(
    spec: JobSpec,
    df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Compile and verify a JobSpec against an actual DataFrame.

    Checks:
    - Target column exists in df.
    - All feature columns exist in df.
    - Split column exists in df (if specified).
    - Target column contains no null values.
    - For classification: target has >= 2 classes, and every class has >= 2 samples.
    - For regression: target must be numeric.

    Returns:
        (is_valid, list_of_error_strings)
    """
    errors: list[str] = []

    # 1. Target column exists in df
    if spec.target_column not in df.columns:
        errors.append(f"Target column '{spec.target_column}' does not exist in dataframe.")
    else:
        # 3. Target column contains no null values
        target_series = df[spec.target_column]
        null_count = int(target_series.isna().sum())
        if null_count > 0:
            errors.append(
                f"Target column '{spec.target_column}' contains {null_count} null values. "
                "TabPFN requires clean targets without missing values."
            )

        # 4. Type & distribution checks based on task_type
        if spec.task_type == "classification":
            non_null_target = target_series.dropna()
            classes = non_null_target.unique()
            if len(classes) < 2:
                errors.append(
                    f"Classification target '{spec.target_column}' must contain at least 2 classes, "
                    f"found {len(classes)} ({list(classes)})."
                )
            else:
                counts = non_null_target.value_counts()
                sparse_classes = counts[counts < 2]
                if not sparse_classes.empty:
                    sparse_dict = sparse_classes.to_dict()
                    errors.append(
                        f"Every class must have at least 2 samples for holdout splitting. "
                        f"Classes with < 2 samples: {sparse_dict}."
                    )
        elif spec.task_type == "regression":
            if not pd.api.types.is_numeric_dtype(target_series):
                errors.append(
                    f"Regression target '{spec.target_column}' must be numeric, "
                    f"but has dtype '{target_series.dtype}'."
                )

    # 2. All feature columns exist in df
    missing_features = [col for col in spec.feature_columns if col not in df.columns]
    if missing_features:
        errors.append(f"Feature columns not found in dataframe: {missing_features}.")

    # Split column exists if specified
    if spec.split_column and spec.split_column not in df.columns:
        errors.append(f"Split column '{spec.split_column}' does not exist in dataframe.")

    is_valid = len(errors) == 0
    return is_valid, errors
